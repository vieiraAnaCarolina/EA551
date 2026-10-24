#ifndef MATMUL_H
#define MATMUL_H

#include <stdint.h>

typedef struct {
    uint32_t rows;
    uint32_t cols;
    double *data;
} Matrix;

typedef struct {
    char magic[4];
    uint32_t rows;
    uint32_t cols;
    uint8_t dtype;
} __attribute__((packed)) EmatHeader;

Matrix *allocate_matrix(uint32_t rows, uint32_t cols);
void free_matrix(Matrix *mat);
Matrix *read_emat(const char *filename);
Matrix *multiply_matrices(const Matrix *A, const Matrix *B);
int compare_matrices(const Matrix *M1, const Matrix *M2, double tolerance);

#endif