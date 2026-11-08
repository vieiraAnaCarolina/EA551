#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <semaphore.h>
#include <dirent.h>
#include <pthread.h>
#include <errno.h>

#include "shm_common.h"

// =========================================================================
// 1. CONSTANTES E DEFINIÇÃO DE ESTRUTURAS 
// =========================================================================
#define SHM_NAME        "/ea551_mnist_shm"
#define SEM_EMPTY_NAME  "/ea551_sem_empty"
#define SEM_FULL_NAME   "/ea551_sem_full"
#define SEM_MUTEX_NAME  "/ea551_sem_mutex"

#define BUFFER_SIZE     256
#define IMAGE_SIZE      784

/*
typedef struct {
    float pixels[IMAGE_SIZE];
    int32_t label;
} SharedSample;

typedef struct {
    SharedSample samples[BUFFER_SIZE];
    int32_t write_idx;
    int32_t read_idx;
    int32_t is_done;
} SharedBuffer;
*/

typedef struct __attribute__((packed)) {
    char magic[4];     // "EMAT"
    uint8_t dtype;     // 0=UINT8, 1=INT32, 2=FLOAT32
    uint32_t rows;
    uint32_t cols;
} EmatHeader;

typedef struct {
    char filepath[1024];
    int label;
} FileTask;

// =========================================================================
// 2. VARIÁVEIS GLOBAIS 
// =========================================================================
static FileTask *file_tasks = NULL;
static size_t total_files = 0;
static size_t current_file_idx = 0;

static pthread_mutex_t task_mutex = PTHREAD_MUTEX_INITIALIZER;
static SharedBuffer *shm_ptr = NULL;
static sem_t *sem_empty = NULL;
static sem_t *sem_full = NULL;
static sem_t *sem_mutex = NULL;

// Novas variáveis globais para controle das worker threads pela stop_dataloader
static pthread_t *worker_threads = NULL;
static int global_num_threads = 0;
static int shm_fd_global = -1;

// =========================================================================
// 3. FUNÇÕES AUXILIARES
// =========================================================================
static ssize_t read_all(int fd, void *buf, size_t count) {
    size_t total = 0;
    char *p = (char *)buf;
    while (total < count) {
        ssize_t n = read(fd, p + total, count - total);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) break;
        total += n;
    }
    return total;
}

static void collect_files(const char *base_path) {
    for (int digit = 0; digit <= 9; digit++) {
        char dir_path[1024];
        snprintf(dir_path, sizeof(dir_path), "%s/%d", base_path, digit);
        
        DIR *dir = opendir(dir_path);
        if (!dir) continue;

        struct dirent *entry;
        while ((entry = readdir(dir)) != NULL) {
            if (strstr(entry->d_name, ".emat")) {
                file_tasks = realloc(file_tasks, sizeof(FileTask) * (total_files + 1));
                snprintf(file_tasks[total_files].filepath, sizeof(file_tasks[total_files].filepath), "%s/%s", dir_path, entry->d_name);
                file_tasks[total_files].label = digit;
                total_files++;
            }
        }
        closedir(dir);
    }
}

static void shuffle_files(FileTask *array, size_t n) {
    if (n > 1) {
        for (size_t i = 0; i < n - 1; i++) {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            FileTask t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

static void *worker_thread(void *arg) {
    while (1) {
        size_t idx;
        pthread_mutex_lock(&task_mutex);
        if (current_file_idx >= total_files) {
            pthread_mutex_unlock(&task_mutex);
            break;
        }
        idx = current_file_idx++;
        pthread_mutex_unlock(&task_mutex);

        int fd = open(file_tasks[idx].filepath, O_RDONLY);
        if (fd < 0) continue;

        EmatHeader header;
        if (read_all(fd, &header, sizeof(EmatHeader)) != sizeof(EmatHeader)) {
            close(fd);
            continue;
        }

        float temp_pixels[IMAGE_SIZE];
        if (header.dtype == 0) { // UINT8
            uint8_t raw[IMAGE_SIZE];
            read_all(fd, raw, IMAGE_SIZE);
            for (int i = 0; i < IMAGE_SIZE; i++) temp_pixels[i] = (float)raw[i];
        } else if (header.dtype == 2) { // FLOAT32
            read_all(fd, temp_pixels, sizeof(float) * IMAGE_SIZE);
        }
        close(fd);

        sem_wait(sem_empty);
        sem_wait(sem_mutex);

        int slot = shm_ptr->write_idx;
        memcpy(shm_ptr->samples[slot].pixels, temp_pixels, sizeof(float) * IMAGE_SIZE);
        shm_ptr->samples[slot].label = file_tasks[idx].label;
        shm_ptr->write_idx = (shm_ptr->write_idx + 1) % BUFFER_SIZE;

        sem_post(sem_mutex);
        sem_post(sem_full);
    }
    return NULL;
}

// =========================================================================
// 4. FUNÇÕES EXPORTADAS PARA A LIBMNIST.SO
// =========================================================================

__attribute__((visibility("default"))) void start_dataloader(int num_threads, const char *emat_dir) {
    if (!emat_dir) emat_dir = "../mnist_emat/train";
    if (num_threads <= 0) num_threads = 4;

    global_num_threads = num_threads;
    total_files = 0;
    current_file_idx = 0;

    // Varre o diretório e embaralha os arquivos
    collect_files(emat_dir);
    shuffle_files(file_tasks, total_files);

    // Criação da SHM e limpeza prévia de recursos residuais
    shm_unlink(SHM_NAME);
    sem_unlink(SEM_EMPTY_NAME);
    sem_unlink(SEM_FULL_NAME); 
    sem_unlink(SEM_MUTEX_NAME);

    shm_fd_global = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (ftruncate(shm_fd_global, sizeof(SharedBuffer)) == -1) {
        perror("ftruncate");
        return;
    }
    shm_ptr = mmap(0, sizeof(SharedBuffer), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_global, 0);
    memset(shm_ptr, 0, sizeof(SharedBuffer));

    // Inicialização dos semáforos POSIX
    sem_empty = sem_open(SEM_EMPTY_NAME, O_CREAT, 0666, BUFFER_SIZE);
    sem_full  = sem_open(SEM_FULL_NAME, O_CREAT, 0666, 0);
    sem_mutex = sem_open(SEM_MUTEX_NAME, O_CREAT, 0666, 1);

    // Aloca e dispara o pool de worker threads nativas em C
    worker_threads = malloc(sizeof(pthread_t) * global_num_threads);
    for (int i = 0; i < global_num_threads; i++) {
        pthread_create(&worker_threads[i], NULL, worker_thread, NULL);
    }
}

__attribute__((visibility("default"))) void stop_dataloader(void) {
    // 1. Aguarda a finalização das worker threads ativas
    if (worker_threads) {
        for (int i = 0; i < global_num_threads; i++) {
            pthread_join(worker_threads[i], NULL);
        }
        free(worker_threads);
        worker_threads = NULL;
    }

    // 2. Sinaliza término no cabeçalho da SHM e destrava eventuais leituras
    if (shm_ptr != NULL) {
        shm_ptr->is_done = 1;
        if (sem_full) sem_post(sem_full);
        
        munmap(shm_ptr, sizeof(SharedBuffer));
        shm_ptr = NULL;
    }

    if (shm_fd_global != -1) {
        close(shm_fd_global);
        shm_fd_global = -1;
    }

    // 3. Libera o vetor de tarefas de arquivos
    if (file_tasks) {
        free(file_tasks);
        file_tasks = NULL;
    }

    // 4. Fecha e remove os semáforos e o segmento SHM do kernel do Linux
    if (sem_empty) sem_close(sem_empty);
    if (sem_full)  sem_close(sem_full);
    if (sem_mutex) sem_close(sem_mutex);

    sem_unlink(SEM_EMPTY_NAME);
    sem_unlink(SEM_FULL_NAME);
    sem_unlink(SEM_MUTEX_NAME);
    shm_unlink(SHM_NAME);
}