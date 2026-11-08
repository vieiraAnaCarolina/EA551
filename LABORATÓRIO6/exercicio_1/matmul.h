#ifndef MATMUL_H
#define MATMUL_H

#include <stdint.h>
#include "shm_common.h"

typedef struct {
    uint32_t rows;
    uint32_t cols;
    double **data; // Ponteiro de linhas apontando para bloco contíguo
} Matrix;

typedef struct __attribute__((packed)) {
    char magic[4];     // "EMAT"
    uint8_t dtype;     // 0=UINT8, 1=INT32, 2=FLOAT32, 3=DOUBLE
    uint32_t rows;
    uint32_t cols;
} EmatHeader;

Matrix *allocate_matrix(uint32_t rows, uint32_t cols);
void free_matrix(Matrix *mat);
Matrix *read_emat(const char *filename);
Matrix *multiply_matrices(const Matrix *A, const Matrix *B);
int compare_matrices(const Matrix *M1, const Matrix *M2, double tolerance);

#endif