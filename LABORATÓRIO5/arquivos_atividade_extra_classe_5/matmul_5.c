#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include "matmul.h"

#define NUM_THREADS 4
#define TILE_SIZE 64

typedef struct {
    const Matrix *A;
    const Matrix *B;
    Matrix *C;
    uint32_t start_row;
    uint32_t end_row;
} ThreadData;

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static ssize_t read_all(int fd, void *buf, size_t count) {
    size_t total_read = 0;
    char *ptr = (char *)buf;
    while (total_read < count) {
        ssize_t bytes_read = read(fd, ptr + total_read, count - total_read);
        if (bytes_read <= 0) {
            if (bytes_read < 0) return -1;
            break;
        }
        total_read += (size_t)bytes_read;
    }
    return (ssize_t)total_read;
}

Matrix *allocate_matrix(uint32_t rows, uint32_t cols) {
    Matrix *mat = (Matrix *)malloc(sizeof(Matrix));
    if (!mat) return NULL;
    
    mat->rows = rows;
    mat->cols = cols;

    mat->data = (double **)malloc(rows * sizeof(double *));
    if (!mat->data) {
        free(mat);
        return NULL;
    }

    //Aloca todos os elementos da matriz num unico bloco contíguo na memória RAM
    double *block = (double *)calloc((size_t)rows * cols, sizeof(double));
    if (!block) {
        free(mat->data);
        free(mat);
        return NULL;
    }

    //Mapeia cada ponteiro de linha para o offset contíguo correto
    for (uint32_t i = 0; i < rows; i++) {
        mat->data[i] = &block[i * cols];
    }

    return mat;
}

void free_matrix(Matrix *mat) {
    if (!mat) return;
    if (mat->data) {
        if (mat->data[0]) free(mat->data[0]);  //Libera o bloco contíguo de dados
        free(mat->data);                       //Libera o vetor de ponteiros
    }
    free(mat);
}

Matrix *read_emat(const char *filename) {
    int fd = open(filename, O_RDONLY);
    if (fd < 0) {
        perror("Erro ao abrir arquivo .emat para leitura");
        return NULL;
    }

    EmatHeader header;
    if (read_all(fd, &header, sizeof(EmatHeader)) != (ssize_t)sizeof(EmatHeader)) {
        fprintf(stderr, "Erro ao ler cabeçalho do arquivo %s\n", filename);
        close(fd);
        return NULL;
    }

    if (strncmp(header.magic, "EMAT", 4) != 0) {
        fprintf(stderr, "Erro: Mágico inválido no arquivo %s\n", filename);
        close(fd);
        return NULL;
    }

    Matrix *mat = allocate_matrix(header.rows, header.cols);
    if (!mat) {
        close(fd);
        return NULL;
    }

    size_t total_elements = (size_t)header.rows * header.cols;

    if (header.dtype == 0) { // UINT8
        uint8_t *raw_buf = (uint8_t *)malloc(total_elements * sizeof(uint8_t));
        if (!raw_buf || read_all(fd, raw_buf, total_elements * sizeof(uint8_t)) != (ssize_t)(total_elements * sizeof(uint8_t))) {
            perror("Erro ao ler elementos UINT8");
            free(raw_buf); free_matrix(mat); close(fd); return NULL;
        }
        for (uint32_t i = 0; i < header.rows; i++) {
            for (uint32_t j = 0; j < header.cols; j++) {
                mat->data[i][j] = (double)raw_buf[i * header.cols + j];
            }
        }
        free(raw_buf);
    } 
    else if (header.dtype == 1) { // INT32
        int32_t *raw_buf = (int32_t *)malloc(total_elements * sizeof(int32_t));
        if (!raw_buf || read_all(fd, raw_buf, total_elements * sizeof(int32_t)) != (ssize_t)(total_elements * sizeof(int32_t))) {
            perror("Erro ao ler elementos INT32");
            free(raw_buf); free_matrix(mat); close(fd); return NULL;
        }
        for (uint32_t i = 0; i < header.rows; i++) {
            for (uint32_t j = 0; j < header.cols; j++) {
                mat->data[i][j] = (double)raw_buf[i * header.cols + j];
            }
        }
        free(raw_buf);
    } 
    else if (header.dtype == 2) { // FLOAT32
        float *raw_buf = (float *)malloc(total_elements * sizeof(float));
        if (!raw_buf || read_all(fd, raw_buf, total_elements * sizeof(float)) != (ssize_t)(total_elements * sizeof(float))) {
            perror("Erro ao ler elementos FLOAT32");
            free(raw_buf); free_matrix(mat); close(fd); return NULL;
        }
        for (uint32_t i = 0; i < header.rows; i++) {
            for (uint32_t j = 0; j < header.cols; j++) {
                mat->data[i][j] = (double)raw_buf[i * header.cols + j];
            }
        }
        free(raw_buf);
    } 
    else if (header.dtype == 3) { // DOUBLE
        if (read_all(fd, mat->data[0], total_elements * sizeof(double)) != (ssize_t)(total_elements * sizeof(double))) {
            perror("Erro ao ler elementos DOUBLE");
            free_matrix(mat); close(fd); return NULL;
        }
    } 
    else {
        fprintf(stderr, "Erro: dtype %d não suportado\n", header.dtype);
        free_matrix(mat);
        close(fd);
        return NULL;
    }

    close(fd);
    return mat;
}

void *multiply_worker(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    const Matrix *A = data->A;
    const Matrix *B = data->B;
    Matrix *C = data->C;

    uint32_t M = data->end_row;
    uint32_t K = A->cols;
    uint32_t N = B->cols;

    //Reordenação i -> k -> j
    for (uint32_t i0 = data->start_row; i0 < M; i0 += TILE_SIZE) {
        for (uint32_t k0 = 0; k0 < K; k0 += TILE_SIZE) {
            for (uint32_t j0 = 0; j0 < N; j0 += TILE_SIZE) {

                uint32_t i_max = (i0 + TILE_SIZE < M) ? i0 + TILE_SIZE : M;
                uint32_t k_max = (k0 + TILE_SIZE < K) ? k0 + TILE_SIZE : K;
                uint32_t j_max = (j0 + TILE_SIZE < N) ? j0 + TILE_SIZE : N;

                for (uint32_t i = i0; i < i_max; i++) {
                    for (uint32_t k = k0; k < k_max; k++) {
                        double r = A->data[i][k];
                        for (uint32_t j = j0; j < j_max; j++) {
                            C->data[i][j] += r * B->data[k][j];
                        }
                    }
                }

            }
        }
    }

    pthread_exit(NULL);
}

Matrix *multiply_matrices(const Matrix *A, const Matrix *B) {
    if (A->cols != B->rows) return NULL;

    Matrix *C = allocate_matrix(A->rows, B->cols);
    if (!C) return NULL;

    pthread_t threads[NUM_THREADS];
    ThreadData thread_data[NUM_THREADS];

    uint32_t rows_per_thread = A->rows / NUM_THREADS;
    uint32_t extra_rows = A->rows % NUM_THREADS;
    uint32_t current_row = 0;

    for (int t = 0; t < NUM_THREADS; t++) {
        thread_data[t].A = A;
        thread_data[t].B = B;
        thread_data[t].C = C;
        thread_data[t].start_row = current_row;

        thread_data[t].end_row = current_row + rows_per_thread + (t < (int)extra_rows ? 1 : 0);
        current_row = thread_data[t].end_row;

        if (pthread_create(&threads[t], NULL, multiply_worker, &thread_data[t]) != 0) {
            perror("Erro ao criar thread");
            free_matrix(C);
            return NULL;
        }
    }

    for (int t = 0; t < NUM_THREADS; t++) {
        pthread_join(threads[t], NULL);
    }

    return C;
}

int compare_matrices(const Matrix *M1, const Matrix *M2, double tolerance) {
    if (M1->rows != M2->rows || M1->cols != M2->cols) return 0;

    for (uint32_t i = 0; i < M1->rows; i++) {
        for (uint32_t j = 0; j < M1->cols; j++) {
            if (fabs(M1->data[i][j] - M2->data[i][j]) > tolerance) {
                return 0;
            }
        }
    }
    return 1;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Uso: %s <A.emat> <B.emat> [C_esperado.emat]\n", argv[0]);
        return 1;
    }

    printf("\n=== Multiplicação Otimizada (Layout Contíguo + Pthreads) ===\n");
    printf("Lendo A: %s\n", argv[1]);
    Matrix *A = read_emat(argv[1]);
    if (!A) return 1;

    printf("Lendo B: %s\n", argv[2]);
    Matrix *B = read_emat(argv[2]);
    if (!B) {
        free_matrix(A);
        return 1;
    }

    printf("Calculando C = A (%ux%u) * B (%ux%u) com %d threads...\n", A->rows, A->cols, B->rows, B->cols, NUM_THREADS);
    
    double start_time = get_time_sec();
    Matrix *C = multiply_matrices(A, B);
    double end_time = get_time_sec();

    if (!C) {
        printf("[RESULTADO] Falha na multiplicação.\n");
        free_matrix(A);
        free_matrix(B);
        return 1;
    }

    printf("[SUCESSO] Tempo de execução da multiplicação: %.6f segundos\n", end_time - start_time);

    if (argc >= 4) {
        printf("Comparando com o gabarito C esperado: %s\n", argv[3]);
        Matrix *C_esperado = read_emat(argv[3]);
        if (C_esperado) {
            if (compare_matrices(C, C_esperado, 1e-4)) {
                printf("[VERIFICAÇÃO OK] O resultado C é IDÊNTICO ao C_esperado!\n");
            } else {
                printf("[VERIFICAÇÃO ERRO] O resultado C difere do C_esperado.\n");
            }
            free_matrix(C_esperado);
        }
    }

    free_matrix(A);
    free_matrix(B);
    free_matrix(C);

    return 0;
}