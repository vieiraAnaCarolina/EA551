#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <semaphore.h>

#include "shm_common.h"
extern void verificar_api_key(void);
#include "fenster.h"

#ifndef TILE_SIZE
#define TILE_SIZE 64
#endif

extern void start_dataloader(int num_threads, const char *emat_dir);
extern void stop_dataloader(void);

typedef struct __attribute__((packed)) {
    char magic[4];
    uint8_t dtype;
    uint32_t rows;
    uint32_t cols;
} EmatHeader;

typedef struct {
    const float *weights;
    const float *bias;
    SharedBuffer *shm_ptr;
    sem_t *sem_empty;
    sem_t *sem_full;
    sem_t *sem_mutex;
    int buf_cap;
    int num_samples;
    int matmul_threads;

    // Controle compartilhado de leitura/despacho e métricas
    pthread_mutex_t *queue_mutex;
    int *read_idx;
    int *dispatched_samples;

    pthread_mutex_t *stats_mutex;
    int *global_correct;
    int *global_errors;
} ConsumerArgs;

typedef struct {
    const float *weights;   // [NUM_CLASSES x IMAGE_SIZE]
    const float *bias;      // [NUM_CLASSES]
    const float *X_batch;   // [N x IMAGE_SIZE]
    float *scores;          // [NUM_CLASSES x N]
    int N;
    int start_row;
    int end_row;
} MatmulThreadArgs;

// =========================================================================
// 1. MULTIPLICAÇÃO UNIFICADA (i -> k -> j + TILING + PTHREADS POR LINHAS)
//    W: [10 x 16384], X_batch: [N x 16384] -> Scores: [10 x N]
// =========================================================================
static void *matmul_rows_worker(void *arg) {
    MatmulThreadArgs *m = (MatmulThreadArgs *)arg;
    const float *weights = m->weights;
    const float *bias    = m->bias;
    const float *X_batch = m->X_batch;
    float *scores        = m->scores;
    int N                = m->N;

    for (int i = m->start_row; i < m->end_row; i++) {
        float b = (bias != NULL) ? bias[i] : 0.0f;
        float *s_row = &scores[i * N];
        for (int j = 0; j < N; j++) {
            s_row[j] = b;
        }
    }

    // Reordenação i -> k -> j com blocagem (tiling) de tamanho TILE_SIZE
    for (int i0 = m->start_row; i0 < m->end_row; i0 += TILE_SIZE) {
        int i_max = (i0 + TILE_SIZE < m->end_row) ? (i0 + TILE_SIZE) : m->end_row;

        for (int k0 = 0; k0 < IMAGE_SIZE; k0 += TILE_SIZE) {
            int k_max = (k0 + TILE_SIZE < IMAGE_SIZE) ? (k0 + TILE_SIZE) : IMAGE_SIZE;

            for (int j0 = 0; j0 < N; j0 += TILE_SIZE) {
                int j_max = (j0 + TILE_SIZE < N) ? (j0 + TILE_SIZE) : N;

                for (int i = i0; i < i_max; i++) {
                    const float *w_row = &weights[i * IMAGE_SIZE];
                    float *s_row = &scores[i * N];

                    for (int k = k0; k < k_max; k++) {
                        float w_ik = w_row[k];
                        for (int j = j0; j < j_max; j++) {
                            s_row[j] += w_ik * X_batch[j * IMAGE_SIZE + k];
                        }
                    }
                }
            }
        }
    }

    return NULL;
}

static void matmul_scores_threaded(
    const float *weights,   // 10 x 16384
    const float *bias,      // 10
    const float *X_batch,   // N x 16384
    float *scores,          // 10 x N
    int N,
    int matmul_threads
) {
    if (matmul_threads <= 1) {
        MatmulThreadArgs single_arg = {
            .weights = weights,
            .bias = bias,
            .X_batch = X_batch,
            .scores = scores,
            .N = N,
            .start_row = 0,
            .end_row = NUM_CLASSES
        };
        matmul_rows_worker(&single_arg);
        return;
    }

    int effective_threads = (matmul_threads > NUM_CLASSES) ? NUM_CLASSES : matmul_threads;
    pthread_t *threads = (pthread_t *)malloc(sizeof(pthread_t) * effective_threads);
    MatmulThreadArgs *targs = (MatmulThreadArgs *)malloc(sizeof(MatmulThreadArgs) * effective_threads);

    if (!threads || !targs) {
        free(threads);
        free(targs);
        MatmulThreadArgs fallback_arg = {
            .weights = weights,
            .bias = bias,
            .X_batch = X_batch,
            .scores = scores,
            .N = N,
            .start_row = 0,
            .end_row = NUM_CLASSES
        };
        matmul_rows_worker(&fallback_arg);
        return;
    }

    int rows_per_thread = NUM_CLASSES / effective_threads;
    int extra_rows = NUM_CLASSES % effective_threads;
    int current_row = 0;

    for (int t = 0; t < effective_threads; t++) {
        targs[t].weights = weights;
        targs[t].bias = bias;
        targs[t].X_batch = X_batch;
        targs[t].scores = scores;
        targs[t].N = N;
        targs[t].start_row = current_row;
        targs[t].end_row = current_row + rows_per_thread + (t < extra_rows ? 1 : 0);
        current_row = targs[t].end_row;

        pthread_create(&threads[t], NULL, matmul_rows_worker, &targs[t]);
    }

    for (int t = 0; t < effective_threads; t++) {
        pthread_join(threads[t], NULL);
    }

    free(threads);
    free(targs);
}

__attribute__((visibility("default"))) int predict_single_sample(
    const float *pixels,
    const float *weights,
    const float *bias
) {
    float scores[NUM_CLASSES];
    matmul_scores_threaded(weights, bias, pixels, scores, 1, 1);

    int best_class = 0;
    float max_score = scores[0];
    for (int c = 1; c < NUM_CLASSES; c++) {
        if (scores[c] > max_score) {
            max_score = scores[c];
            best_class = c;
        }
    }
    return best_class;
}

// =========================================================================
// 2. THREAD CONSUMIDORA NO MODO TESTE
// =========================================================================
static void *consumer_worker(void *arg) {
    ConsumerArgs *ctx = (ConsumerArgs *)arg;
    const int BATCH = 64;

    float *X_batch = (float *)malloc(sizeof(float) * BATCH * IMAGE_SIZE);
    float *scores  = (float *)malloc(sizeof(float) * NUM_CLASSES * BATCH);
    int   *labels  = (int *)malloc(sizeof(int) * BATCH);

    if (!X_batch || !scores || !labels) {
        free(X_batch);
        free(scores);
        free(labels);
        return NULL;
    }

    int local_correct = 0;
    int local_errors = 0;

    while (1) {
        int cur_batch = 0;

        for (int b = 0; b < BATCH; b++) {
            pthread_mutex_lock(ctx->queue_mutex);
            if (*ctx->dispatched_samples >= ctx->num_samples) {
                pthread_mutex_unlock(ctx->queue_mutex);
                break;
            }
            (*ctx->dispatched_samples)++;
            pthread_mutex_unlock(ctx->queue_mutex);

            sem_wait(ctx->sem_full);
            sem_wait(ctx->sem_mutex);

            int idx = *ctx->read_idx;
            memcpy(&X_batch[cur_batch * IMAGE_SIZE], ctx->shm_ptr->samples[idx].pixels, sizeof(float) * IMAGE_SIZE);
            labels[cur_batch] = ctx->shm_ptr->samples[idx].label;
            *ctx->read_idx = (idx + 1) % ctx->buf_cap;

            sem_post(ctx->sem_mutex);
            sem_post(ctx->sem_empty);

            cur_batch++;
        }

        if (cur_batch == 0) {
            break;
        }

        matmul_scores_threaded(ctx->weights, ctx->bias, X_batch, scores, cur_batch, ctx->matmul_threads);

        for (int b = 0; b < cur_batch; b++) {
            int best_c = 0;
            float best_s = scores[0 * cur_batch + b];
            for (int c = 1; c < NUM_CLASSES; c++) {
                float s = scores[c * cur_batch + b];
                if (s > best_s) {
                    best_s = s;
                    best_c = c;
                }
            }
            if (best_c == labels[b]) {
                local_correct++;
            } else {
                local_errors++;
            }
        }
    }

    pthread_mutex_lock(ctx->stats_mutex);
    *ctx->global_correct += local_correct;
    *ctx->global_errors  += local_errors;
    pthread_mutex_unlock(ctx->stats_mutex);

    free(X_batch);
    free(scores);
    free(labels);
    return NULL;
}

// =========================================================================
// 3. MODO TESTE COM MÚLTIPLOS CONSUMIDORES + DATALOADER + MATMUL UNIFICADO
// =========================================================================
__attribute__((visibility("default"))) float run_test_from_dataloader(
    const float *weights,
    const float *bias,
    const char *test_dir,
    int num_threads,
    int num_samples,
    int buffer_size,
    int num_consumers,
    int matmul_threads
) {
    verificar_api_key();
    if (num_samples <= 0) num_samples = 10000;
    if (num_consumers <= 0) num_consumers = 1;
    if (matmul_threads <= 0) matmul_threads = 1;
    int buf_cap = (buffer_size > 0 && buffer_size <= BUFFER_SIZE) ? buffer_size : BUFFER_SIZE;

    start_dataloader(num_threads, test_dir);

    int shm_fd = shm_open(SHM_NAME, O_RDWR, 0666);
    if (shm_fd < 0) {
        perror("shm_open no teste");
        stop_dataloader();
        return 0.0f;
    }

    SharedBuffer *shm_ptr = mmap(0, sizeof(SharedBuffer), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    sem_t *sem_empty = sem_open(SEM_EMPTY_NAME, 0);
    sem_t *sem_full  = sem_open(SEM_FULL_NAME, 0);
    sem_t *sem_mutex = sem_open(SEM_MUTEX_NAME, 0);

    pthread_mutex_t queue_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t stats_mutex = PTHREAD_MUTEX_INITIALIZER;

    int read_idx = 0;
    int dispatched_samples = 0;
    int global_correct = 0;
    int global_errors = 0;

    pthread_t *consumers = (pthread_t *)malloc(sizeof(pthread_t) * num_consumers);
    ConsumerArgs *args = (ConsumerArgs *)malloc(sizeof(ConsumerArgs) * num_consumers);

    for (int i = 0; i < num_consumers; i++) {
        args[i].weights = weights;
        args[i].bias = bias;
        args[i].shm_ptr = shm_ptr;
        args[i].sem_empty = sem_empty;
        args[i].sem_full = sem_full;
        args[i].sem_mutex = sem_mutex;
        args[i].buf_cap = buf_cap;
        args[i].num_samples = num_samples;
        args[i].matmul_threads = matmul_threads;
        args[i].queue_mutex = &queue_mutex;
        args[i].read_idx = &read_idx;
        args[i].dispatched_samples = &dispatched_samples;
        args[i].stats_mutex = &stats_mutex;
        args[i].global_correct = &global_correct;
        args[i].global_errors = &global_errors;

        pthread_create(&consumers[i], NULL, consumer_worker, &args[i]);
    }

    for (int i = 0; i < num_consumers; i++) {
        pthread_join(consumers[i], NULL);
    }

    free(consumers);
    free(args);
    pthread_mutex_destroy(&queue_mutex);
    pthread_mutex_destroy(&stats_mutex);

    munmap(shm_ptr, sizeof(SharedBuffer));
    close(shm_fd);
    sem_close(sem_empty);
    sem_close(sem_full);
    sem_close(sem_mutex);

    stop_dataloader();

    int total_evaluated = global_correct + global_errors;
    float accuracy = (total_evaluated > 0) ? ((float)global_correct / (float)total_evaluated) * 100.0f : 0.0f;

    printf("[C - Matmul Test] Amostras: %d | Produtores: %d | Consumidores: %d | MatmulThreads: %d | Buffer: %d | Acertos: %d | Erros: %d | Acurácia: %.2f%%\n",
           total_evaluated, num_threads, num_consumers, matmul_threads, buf_cap, global_correct, global_errors, accuracy);
    return accuracy;
}

// =========================================================================
// 4. MODO INFERÊNCIA COM FORK() + JANELA 128x128 VIA FENSTER.H
// =========================================================================
static void show_fenster_128x128(const float *pixels, int predicted_digit) {
    uint32_t buf[IMAGE_SIZE];
    for (int i = 0; i < IMAGE_SIZE; i++) {
        float v = pixels[i];
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        uint32_t gray = (uint32_t)(v * 255.0f);
        buf[i] = (gray << 16) | (gray << 8) | gray;
    }

    char title[128];
    snprintf(title, sizeof(title), "EA551 Inferencia (128x128) - Predicao: %d [ESC/Q para sair]", predicted_digit);

    struct fenster f = {
        .title = title,
        .width = IMAGE_DIM,
        .height = IMAGE_DIM,
        .buf = buf
    };

    if (fenster_open(&f) == 0) {
        while (fenster_loop(&f) == 0) {
            if (f.keys[27] || f.keys['Q'] || f.keys['q']) break;
            fenster_sleep(16);
        }
        fenster_close(&f);
    }
}

__attribute__((visibility("default"))) void run_inference_file(
    const char *emat_path,
    const float *weights,
    const float *bias,
    int enable_x11
) {
    verificar_api_key();
    pid_t pid = fork();
    if (pid == 0) {
        int fd = open(emat_path, O_RDONLY);
        if (fd < 0) {
            perror("[Filho - Infer] Erro ao abrir imagem .emat");
            exit(1);
        }

        EmatHeader header;
        if (read(fd, &header, sizeof(EmatHeader)) != sizeof(EmatHeader) || memcmp(header.magic, "EMAT", 4) != 0) {
            fprintf(stderr, "[Filho - Infer] Cabeçalho .emat inválido em %s\n", emat_path);
            close(fd);
            exit(1);
        }

        float *pixels = (float *)malloc(sizeof(float) * IMAGE_SIZE);
        if (header.dtype == 0) {
            uint8_t *raw = (uint8_t *)malloc(IMAGE_SIZE);
            if (read(fd, raw, IMAGE_SIZE) != IMAGE_SIZE) { close(fd); exit(1); }
            for (int i = 0; i < IMAGE_SIZE; i++) pixels[i] = (float)raw[i] / 255.0f;
            free(raw);
        } else {
            if (read(fd, pixels, sizeof(float) * IMAGE_SIZE) != (ssize_t)(sizeof(float) * IMAGE_SIZE)) { close(fd); exit(1); }
        }
        close(fd);

        int predicted_digit = predict_single_sample(pixels, weights, bias);
        printf("[Processo Filho PID=%d] Imagem: %s -> Classe Predita: %d\n", getpid(), emat_path, predicted_digit);
        fflush(stdout);

        if (enable_x11) {
            show_fenster_128x128(pixels, predicted_digit);
        }

        free(pixels);
        exit(0);
    } else if (pid > 0) {
        waitpid(pid, NULL, 0);
    } else {
        perror("fork");
    }
}