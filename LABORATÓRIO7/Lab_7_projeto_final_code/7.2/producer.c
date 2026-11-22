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

typedef struct __attribute__((packed)) {
    char magic[4];
    uint8_t dtype;
    uint32_t rows;
    uint32_t cols;
} EmatHeader;

typedef struct {
    char filepath[1024];
    int label;
} FileTask;

static FileTask *file_tasks = NULL;
static size_t total_files = 0;
static size_t current_file_idx = 0;
static volatile int stop_requested = 0;

static pthread_mutex_t task_mutex = PTHREAD_MUTEX_INITIALIZER;
static SharedBuffer *shm_ptr = NULL;
static sem_t *sem_empty = NULL;
static sem_t *sem_full = NULL;
static sem_t *sem_mutex = NULL;

static pthread_t *worker_threads = NULL;
static int global_num_threads = 0;
static int shm_fd_global = -1;

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
    (void)arg;
    float *temp_pixels = (float *)malloc(sizeof(float) * IMAGE_SIZE);
    uint8_t *raw_u8 = (uint8_t *)malloc(sizeof(uint8_t) * IMAGE_SIZE);
    if (!temp_pixels || !raw_u8) {
        free(temp_pixels);
        free(raw_u8);
        return NULL;
    }

    while (!stop_requested) {
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

        size_t n_elems = (size_t)header.rows * header.cols;
        if (n_elems != IMAGE_SIZE) {
            close(fd);
            continue;
        }

        if (header.dtype == 0) {
            if (read_all(fd, raw_u8, IMAGE_SIZE) != IMAGE_SIZE) { close(fd); continue; }
            for (int i = 0; i < IMAGE_SIZE; i++) temp_pixels[i] = (float)raw_u8[i] / 255.0f;
        } else if (header.dtype == 2) {
            if (read_all(fd, temp_pixels, sizeof(float) * IMAGE_SIZE) != (ssize_t)(sizeof(float) * IMAGE_SIZE)) { close(fd); continue; }
        }
        close(fd);

        sem_wait(sem_empty);
        if (stop_requested) {
            sem_post(sem_empty);
            break;
        }
        sem_wait(sem_mutex);

        int slot = shm_ptr->write_idx;
        memcpy(shm_ptr->samples[slot].pixels, temp_pixels, sizeof(float) * IMAGE_SIZE);
        shm_ptr->samples[slot].label = file_tasks[idx].label;
        shm_ptr->write_idx = (shm_ptr->write_idx + 1) % BUFFER_SIZE;

        sem_post(sem_mutex);
        sem_post(sem_full);
    }

    free(temp_pixels);
    free(raw_u8);
    return NULL;
}

__attribute__((visibility("default"))) void start_dataloader(int num_threads, const char *emat_dir) {
    if (!emat_dir) emat_dir = "../../big_mnist_emat/train";
    if (num_threads <= 0) num_threads = 4;

    global_num_threads = num_threads;
    total_files = 0;
    current_file_idx = 0;
    stop_requested = 0;

    collect_files(emat_dir);
    shuffle_files(file_tasks, total_files);

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

    sem_empty = sem_open(SEM_EMPTY_NAME, O_CREAT, 0666, BUFFER_SIZE);
    sem_full  = sem_open(SEM_FULL_NAME, O_CREAT, 0666, 0);
    sem_mutex = sem_open(SEM_MUTEX_NAME, O_CREAT, 0666, 1);

    worker_threads = malloc(sizeof(pthread_t) * global_num_threads);
    for (int i = 0; i < global_num_threads; i++) {
        pthread_create(&worker_threads[i], NULL, worker_thread, NULL);
    }
}

__attribute__((visibility("default"))) void stop_dataloader(void) {
    stop_requested = 1;
    if (sem_empty) {
        for (int i = 0; i < global_num_threads + BUFFER_SIZE; i++) sem_post(sem_empty);
    }

    if (worker_threads) {
        for (int i = 0; i < global_num_threads; i++) {
            pthread_join(worker_threads[i], NULL);
        }
        free(worker_threads);
        worker_threads = NULL;
    }

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

    if (file_tasks) {
        free(file_tasks);
        file_tasks = NULL;
    }

    if (sem_empty) sem_close(sem_empty);
    if (sem_full)  sem_close(sem_full);
    if (sem_mutex) sem_close(sem_mutex);

    sem_unlink(SEM_EMPTY_NAME);
    sem_unlink(SEM_FULL_NAME);
    sem_unlink(SEM_MUTEX_NAME);
    shm_unlink(SHM_NAME);
}
