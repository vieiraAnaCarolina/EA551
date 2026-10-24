#ifndef SHM_COMMON_H
#define SHM_COMMON_H

#include <stdint.h>

#define SHM_NAME "/ea551_mnist_shm"
#define SEM_EMPTY_NAME "/ea551_sem_empty"
#define SEM_FULL_NAME "/ea551_sem_full"
#define SEM_MUTEX_NAME "/ea551_sem_mutex"

#define BUFFER_SIZE 256
#define IMAGE_SIZE 784 // 28x28

typedef struct {
    float pixels[IMAGE_SIZE];
    int label;
} Sample;

typedef struct {
    Sample samples[BUFFER_SIZE];
    int write_idx;
    int read_idx;
    int is_done;
} SharedBuffer;

#endif