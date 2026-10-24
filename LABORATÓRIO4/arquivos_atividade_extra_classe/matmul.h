#ifndef MATMUL_H
#define MATMUL_H
#include <stdint.h>
#include <stddef.h>

// Estrutura do cabeçalho binário do .emat (13 bytes)
typedef struct __attribute__((packed)) {
    char magic[4];     // EMAT
    uint8_t dtype;     // Tipo de dado (0=UINT8, 1=INT32, 2=FLOAT32, 3=DOUBLE)
    uint32_t rows;     // Número de linhas
    uint32_t cols;     // Número de colunas
} EmatHeader;

// Estrutura de representação da Matriz usando double **
typedef struct {
    uint32_t rows;
    uint32_t cols;
    double **data;
} Matrix;


Matrix *allocate_matrix(uint32_t rows, uint32_t cols);
void free_matrix(Matrix *mat);
Matrix *read_emat(const char *filename);
int write_emat(const char *filename, const Matrix *mat);
Matrix *multiply_matrices(const Matrix *A, const Matrix *B);
int compare_matrices(const Matrix *M1, const Matrix *M2, double tolerance);

#endif 