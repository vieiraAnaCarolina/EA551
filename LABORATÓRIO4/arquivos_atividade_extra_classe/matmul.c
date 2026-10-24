#include "matmul.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <math.h>
#include <string.h>

// Leitura segura completa com syscall read
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

// Escrita segura completa com syscall write
static ssize_t write_all(int fd, const void *buf, size_t count) {
    size_t total = 0;
    const char *p = (const char *)buf;
    while (total < count) {
        ssize_t n = write(fd, p + total, count - total);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        total += n;
    }
    return total;
}

// Alocação dinâmica para double **
Matrix *allocate_matrix(uint32_t rows, uint32_t cols) {
    Matrix *mat = (Matrix *)malloc(sizeof(Matrix));
    if (!mat) {
        perror("Erro ao alocar estrutura Matrix");
        return NULL;
    }
    mat->rows = rows;
    mat->cols = cols;

    mat->data = (double **)malloc(rows * sizeof(double *));
    if (!mat->data) {
        perror("Erro ao alocar ponteiros de linhas");
        free(mat);
        return NULL;
    }

    for (uint32_t i = 0; i < rows; i++) {
        mat->data[i] = (double *)calloc(cols, sizeof(double));
        if (!mat->data[i]) {
            perror("Erro ao alocar colunas da matriz");
            for (uint32_t j = 0; j < i; j++) free(mat->data[j]);
            free(mat->data);
            free(mat);
            return NULL;
        }
    }
    return mat;
}

// Desalocação da matriz double **
void free_matrix(Matrix *mat) {
    if (!mat) return;
    if (mat->data) {
        for (uint32_t i = 0; i < mat->rows; i++) {
            if (mat->data[i]) free(mat->data[i]);
        }
        free(mat->data);
    }
    free(mat);
}

// Leitura do arquivo .emat suportando UINT8, INT32, FLOAT32 e DOUBLE
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
        double *raw_buf = (double *)malloc(total_elements * sizeof(double));
        if (!raw_buf || read_all(fd, raw_buf, total_elements * sizeof(double)) != (ssize_t)(total_elements * sizeof(double))) {
            perror("Erro ao ler elementos DOUBLE");
            free(raw_buf); free_matrix(mat); close(fd); return NULL;
        }
        for (uint32_t i = 0; i < header.rows; i++) {
            for (uint32_t j = 0; j < header.cols; j++) {
                mat->data[i][j] = raw_buf[i * header.cols + j];
            }
        }
        free(raw_buf);
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

// Escrita da matriz no arquivo .emat via syscall write
int write_emat(const char *filename, const Matrix *mat) {
    int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        perror("Erro ao abrir arquivo .emat para escrita");
        return -1;
    }

    EmatHeader header = {
        .magic = {'E', 'M', 'A', 'T'},
        .dtype = 3, // DOUBLE
        .rows = mat->rows,
        .cols = mat->cols
    };

    if (write_all(fd, &header, sizeof(EmatHeader)) != (ssize_t)sizeof(EmatHeader)) {
        perror("Erro ao escrever cabeçalho .emat");
        close(fd);
        return -1;
    }

    size_t total_elements = (size_t)mat->rows * mat->cols;
    double *raw_buf = (double *)malloc(total_elements * sizeof(double));
    if (!raw_buf) {
        perror("Erro de alocação no buffer de escrita");
        close(fd);
        return -1;
    }

    for (uint32_t i = 0; i < mat->rows; i++) {
        for (uint32_t j = 0; j < mat->cols; j++) {
            raw_buf[i * mat->cols + j] = mat->data[i][j];
        }
    }

    if (write_all(fd, raw_buf, total_elements * sizeof(double)) != (ssize_t)(total_elements * sizeof(double))) {
        perror("Erro ao escrever dados da matriz no arquivo .emat");
        free(raw_buf);
        close(fd);
        return -1;
    }

    free(raw_buf);
    close(fd);
    return 0;
}

// Multiplicação otimizada i -> k -> j
Matrix *multiply_matrices(const Matrix *A, const Matrix *B) {
    if (A->cols != B->rows) {
        fprintf(stderr, "Erro de dimensão: Colunas de A (%u) != Linhas de B (%u)\n", A->cols, B->rows);
        return NULL;
    }

    Matrix *C = allocate_matrix(A->rows, B->cols);
    if (!C) return NULL;

    for (uint32_t i = 0; i < A->rows; i++) {
        for (uint32_t k = 0; k < A->cols; k++) {
            double r = A->data[i][k];
            for (uint32_t j = 0; j < B->cols; j++) {
                C->data[i][j] += r * B->data[k][j];
            }
        }
    }

    return C;
}

// Comparação elemento a elemento com tolerância de ponto flutuante
int compare_matrices(const Matrix *M1, const Matrix *M2, double tolerance) {
    if (M1->rows != M2->rows || M1->cols != M2->cols) {
        printf("  [DIFERENÇA] Dimensões distintas: (%ux%u) vs (%ux%u)\n", M1->rows, M1->cols, M2->rows, M2->cols);
        return 0;
    }

    for (uint32_t i = 0; i < M1->rows; i++) {
        for (uint32_t j = 0; j < M1->cols; j++) {
            if (fabs(M1->data[i][j] - M2->data[i][j]) > tolerance) {
                printf("  [DIFERENÇA] Na posição [%u][%u]: calculado=%f vs esperado=%f\n", i, j, M1->data[i][j], M2->data[i][j]);
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

    printf("\n=== Multiplicação de Matrizes .emat ===\n");
    printf("Lendo A: %s\n", argv[1]);
    Matrix *A = read_emat(argv[1]);
    if (!A) return 1;

    printf("Lendo B: %s\n", argv[2]);
    Matrix *B = read_emat(argv[2]);
    if (!B) {
        free_matrix(A);
        return 1;
    }

    printf("Calculando C = A (%ux%u) * B (%ux%u)...\n", A->rows, A->cols, B->rows, B->cols);
    Matrix *C = multiply_matrices(A, B);
    if (!C) {
        printf("[RESULTADO] Falha na multiplicação devido a erro de dimensão/alocação.\n");
        free_matrix(A);
        free_matrix(B);
        return 1;
    }

    printf("[SUCESSO] Multiplicação realizada! Matriz C resultante: %ux%u\n", C->rows, C->cols);

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