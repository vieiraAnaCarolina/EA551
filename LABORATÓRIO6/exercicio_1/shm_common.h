#ifndef SHM_COMMON_H
#define SHM_COMMON_H

#include <stdint.h>

#define SHM_NAME        "/ea551_mnist_shm"
#define SEM_EMPTY_NAME  "/ea551_sem_empty"
#define SEM_FULL_NAME   "/ea551_sem_full"
#define SEM_MUTEX_NAME  "/ea551_sem_mutex"

#define BUFFER_SIZE     256
#define IMAGE_SIZE      784

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

#endif // SHM_COMMON_H