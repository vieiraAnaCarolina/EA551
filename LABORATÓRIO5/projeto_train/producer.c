#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
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

typedef struct __attribute__((packed)) {
    char magic[4];     // EMAT
    uint8_t dtype;     // 0=UINT8, 1=INT32, 2=FLOAT32
    uint32_t rows;
    uint32_t cols;
} EmatHeader;

typedef struct {
    char filepath[1024];
    int label;
} FileTask;

FileTask *file_tasks = NULL;
size_t total_files = 0;
size_t current_file_idx = 0;

pthread_mutex_t task_mutex = PTHREAD_MUTEX_INITIALIZER;
SharedBuffer *shm_ptr = NULL;
sem_t *sem_empty = NULL;
sem_t *sem_full = NULL;
sem_t *sem_mutex = NULL;

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

void collect_files(const char *base_path) {
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

void *worker_thread(void *arg) {
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

int main(int argc, char *argv[]) {
    int num_threads = (argc > 1) ? atoi(argv[1]) : 1;
    const char *emat_dir = (argc > 2) ? argv[2] : "../mnist_emat/train";

    printf("Indexando arquivos .emat de %s...\n", emat_dir);
    collect_files(emat_dir);
    printf("Total de amostras encontradas: %zu\n", total_files);

    shm_unlink(SHM_NAME);
    sem_unlink(SEM_EMPTY_NAME);
    sem_unlink(SEM_FULL_NAME);
    sem_unlink(SEM_MUTEX_NAME);

    int shm_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (ftruncate(shm_fd, sizeof(SharedBuffer)) == -1) {
        perror("ftruncate");
        return 1;
    }
    shm_ptr = mmap(0, sizeof(SharedBuffer), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);

    memset(shm_ptr, 0, sizeof(SharedBuffer));

    sem_empty = sem_open(SEM_EMPTY_NAME, O_CREAT, 0666, BUFFER_SIZE);
    sem_full  = sem_open(SEM_FULL_NAME, O_CREAT, 0666, 0);
    sem_mutex = sem_open(SEM_MUTEX_NAME, O_CREAT, 0666, 1);

    pthread_t threads[num_threads];
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, worker_thread, NULL);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    shm_ptr->is_done = 1;
    sem_post(sem_full);

    printf("Produção concluída pelo C.\n");
    return 0;
}