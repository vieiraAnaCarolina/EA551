#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h> // Biblioteca necessária para as system calls read, write, STDIN_FILENO e STDOUT_FILENO
#include <errno.h>

#include "stb_ds.h"
#include "ea551-mat-txt-parser.h"

extern int yyparse(void); // Declarar a função yyparse como externa

// Estrutura do cabeçalho de 13 bytes (sem padding)
typedef struct __attribute__((packed)) {
    char magic[4];     // EMAT
    uint8_t dtype;     // Tipo de dado (0=UINT8, 1=INT32, 2=FLOAT32)
    uint32_t rows;     // Número de linhas 
    uint32_t cols;     // Número de colunas 
} EmatHeader;

// =========================================================================
// FUNÇÕES AUXILIARES COM LAÇO PARA EVITAR LEITURAS/ESCRITAS PARCIAIS
// =========================================================================

// Garante que TODOS os bytes solicitados sejam lidos do descriptor
static ssize_t read_all(int fd, void *buf, size_t count) {
    size_t total_lido = 0;
    char *p = (char *)buf;

    while (total_lido < count) {
        ssize_t n = read(fd, p + total_lido, count - total_lido);
        if (n < 0) {
            if (errno == EINTR) continue; // Tenta novamente se for interrompido por sinal
            return -1;                   // Erro real de leitura
        }
        if (n == 0) break; // Fim de ficheiro (EOF) prematuro
        total_lido += n;
    }
    return total_lido;
}

// Garante que TODOS os bytes solicitados sejam escritos no descriptor
static ssize_t write_all(int fd, const void *buf, size_t count) {
    size_t total_escrito = 0;
    const char *p = (const char *)buf;

    while (total_escrito < count) {
        ssize_t n = write(fd, p + total_escrito, count - total_escrito);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue; // Tenta novamente se for interrompido
            return -1;                            // Erro real de escrita
        }
        total_escrito += n;
    }
    return total_escrito;
}

int main(void)
{
    int c;
    while ((c = getchar()) != EOF && (c == ' ' || c == '\t' || c == '\r' || c == '\n'));
    if (c == EOF) return 1;
    ungetc(c, stdin);

    // =========================================================================
    // CAMINHO 1: MATRIZ EM TEXTO (.txt) -> ARQUIVO BINÁRIO (.emat)
    // =========================================================================
    if (c == '[') {
        yyin = stdin;

        if (yyparse() != 0 || !ea551_mat) {
            return 1;
        }

        uint32_t r = (uint32_t) arrlen(ea551_mat->linhas);
        uint32_t cols = (r > 0) ? (uint32_t) arrlen(ea551_mat->linhas[0]) : 0;

        EmatHeader header = {
            .magic = {'E', 'M', 'A', 'T'},
            .dtype = (uint8_t) ea551_mat->dtype,
            .rows  = r,
            .cols  = cols
        };

        // Escrita do cabeçalho de 13 bytes garantida via laço write_all
        if (write_all(STDOUT_FILENO, &header, sizeof(EmatHeader)) != (ssize_t)sizeof(EmatHeader)) {
            fprintf(stderr, "Erro ao escrever cabeçalho com write_all\n");
            return 1;
        }

        // Escrita dos dados da matriz garantida via laço write_all
        for (uint32_t i = 0; i < r; i++) {
            for (uint32_t j = 0; j < cols; j++) {
                double val = ea551_mat->linhas[i][j];

                if (ea551_mat->dtype == EA551_DTYPE_UINT8) {
                    uint8_t v = (uint8_t) val;
                    if (write_all(STDOUT_FILENO, &v, sizeof(uint8_t)) != (ssize_t)sizeof(uint8_t)) {
                        fprintf(stderr, "Erro ao escrever elemento uint8 com write_all\n");
                        return 1;
                    }
                } else if (ea551_mat->dtype == EA551_DTYPE_INT32) {
                    int32_t v = (int32_t) val;
                    if (write_all(STDOUT_FILENO, &v, sizeof(int32_t)) != (ssize_t)sizeof(int32_t)) {
                        fprintf(stderr, "Erro ao escrever elemento int32 com write_all\n");
                        return 1;
                    }
                } else if (ea551_mat->dtype == EA551_DTYPE_FLOAT32) {
                    float v = (float) val;
                    if (write_all(STDOUT_FILENO, &v, sizeof(float)) != (ssize_t)sizeof(float)) {
                        fprintf(stderr, "Erro ao escrever elemento float32 com write_all\n");
                        return 1;
                    }
                }
            }
        }

        ea551_mat_free(ea551_mat);
        return 0;
    }

    // =========================================================================
    // CAMINHO 2: ARQUIVO BINÁRIO (.emat) -> MATRIZ EM TEXTO (.txt)
    // =========================================================================
    EmatHeader header;

    // Leitura garantida de todos os 13 bytes do cabeçalho via laço read_all
    if (read_all(STDIN_FILENO, &header, sizeof(EmatHeader)) != (ssize_t)sizeof(EmatHeader)) {
        fprintf(stderr, "Erro ao ler o cabeçalho do ficheiro .emat\n");
        return 1;
    }

    // Compara os 4 primeiros bytes do arquivo lido com "EMAT" para validá-lo
    if (memcmp(header.magic, "EMAT", 4) != 0) {
        fprintf(stderr, "Erro: Assinatura 'EMAT' inválida.\n");
        return 1;
    }

    // Reconstrução da matriz textual a partir dos dados lidos
    printf("[\n");
    for (uint32_t i = 0; i < header.rows; i++) {
        printf("  [");
        for (uint32_t j = 0; j < header.cols; j++) {
            if (header.dtype == EA551_DTYPE_UINT8) {
                uint8_t v;
                if (read_all(STDIN_FILENO, &v, sizeof(uint8_t)) != (ssize_t)sizeof(uint8_t)) {
                    fprintf(stderr, "Erro ao ler elemento uint8 com read_all\n");
                    return 1;
                }
                printf("%u", v);
            } else if (header.dtype == EA551_DTYPE_INT32) {
                int32_t v;
                if (read_all(STDIN_FILENO, &v, sizeof(int32_t)) != (ssize_t)sizeof(int32_t)) {
                    fprintf(stderr, "Erro ao ler elemento int32 com read_all\n");
                    return 1;
                }
                printf("%d", v);
            } else if (header.dtype == EA551_DTYPE_FLOAT32) {
                float v;
                if (read_all(STDIN_FILENO, &v, sizeof(float)) != (ssize_t)sizeof(float)) {
                    fprintf(stderr, "Erro ao ler elemento float32 com read_all\n");
                    return 1;
                }
                printf("%g", v);
            }
            
            // Se não for a última coluna, imprime uma vírgula entre os elementos
            if (j < header.cols - 1) printf(", ");
        }
        printf("]");

        // Se não for a última linha da matriz, adiciona vírgula após o colchete
        if (i < header.rows - 1) printf(",");
        printf("\n");
    }
    printf("]\n");

    return 0;
}