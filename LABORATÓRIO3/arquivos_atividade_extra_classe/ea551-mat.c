#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "ea551-mat-txt-parser.h"
extern int yyparse(void); //declarar a variável parse como externa

//VERSÃO ANTIGA DO MAT.C, NÃO USAR ESSA

int main(void)
{
    int c = getchar();
    if (c == EOF) return 1;
    ungetc(c, stdin);

    //Caso temos um arquivo txt
    if (c == '[') {          /* stdin = bytes do .txt */
        yyin = stdin;
        // Chama o parser para interpretar a stream de caracteres de entrada como uma matriz na memória RAM
        // Note que ea551_mat é definido em ea551-mat-txt-parser.h
        if (yyparse() != 0 || !ea551_mat) return 1;
        // Chegamos aqui se o parser rodou sem erros e ea551_mat é um ponteiro válido para a matriz na memória RAM
        print_matrix(ea551_mat);

        //Estrutura do cabeçalho de 13 bytes, considerando todos oa valores sem espaços
        typedef struct __attribute__((packed)) {
            char magic[4];     //EMAT
            uint8_t dtype;     //Tipo
            uint32_t rows;     //Número de linhas 
            uint32_t cols;     //Número de colunas 
        } EmatHeader;

        // Obter dimensões usando stb_ds.h 
        uint32_t r = (uint32_t) arrlen(ea551_mat->linhas); //Quantidade total de linhas na matriz
        //Caso a matriz tenha mais que 1 linha, o arrlen devolve a quantidade de elementos na linha
        // (que corresponde ao número de colunas).
        uint32_t cols = (r > 0) ? (uint32_t) arrlen(ea551_mat->linhas[0]) : 0;

        //Montar o cabeçalho
        EmatHeader header = {
            .magic = {'E', 'M', 'A', 'T'},
            .dtype = (uint8_t) ea551_mat->dtype,
            .rows  = r,
            .cols  = cols
        };

        //Realiza a escrita binária direto na memória onde header foi montada.
        //stdout como destino 
        fwrite(&header, sizeof(EmatHeader), 1, stdout);

        //Escrita dos dados numéricos reais da matriz no arquivo binário após o cabeçalho feito
        //Laço para percorrer as linhas e colunas da matriz
        for (uint32_t i = 0; i < r; i++) {
            for (uint32_t j = 0; j < cols; j++) {
                //Acesso do valor do item em específico pelo double que são guardados pelo parser 
                double val = ea551_mat->linhas[i][j];

                //Parser calcula tipo do valor final, com esse valor a fwrite recebe o tamanho que deverá 
                //ser gravado em bytes por elemento e os escreve em binário.
                if (ea551_mat->dtype == EA551_DTYPE_UINT8) {
                    uint8_t v = (uint8_t) val;
                    fwrite(&v, sizeof(uint8_t), 1, stdout);
                } else if (ea551_mat->dtype == EA551_DTYPE_INT32) {
                    int32_t v = (int32_t) val;
                    fwrite(&v, sizeof(int32_t), 1, stdout);
                } else if (ea551_mat->dtype == EA551_DTYPE_FLOAT32) {
                    float v = (float) val;
                    fwrite(&v, sizeof(float), 1, stdout);
                }
            }
        }

        ea551_mat_free(ea551_mat);
        return 0;
    }

    /* TODO: stdin = bytes do .emat — ler header + sequência de bytes no tipo de ea551_mat->dtype e imprimir a matriz */
    return 1;
}

