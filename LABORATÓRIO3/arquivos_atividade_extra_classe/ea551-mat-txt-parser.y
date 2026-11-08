%{
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "ea551-mat-txt-parser.h"

#define STB_DS_IMPLEMENTATION
#include "stb_ds.h"

emat *ea551_mat; //Variável global que vai armazenar o resultado final da matriz
extern int yylineno;
int yylex(void);
void yyerror(const char *s);

//Função auxiliar que retorna o maior tipo entre dois tipos
//Importante para o processo de promoção de tipos dos dados processados
static ea551_dtype dtype_max(ea551_dtype a, ea551_dtype b)
{
    return a > b ? a : b;
}
%}

%union {
    double dval;
    struct { double valor; ea551_dtype dtype; } valor;       //Representa um único número
    struct { double *valores; ea551_dtype dtype; } linha;    //Representa uma linha de números
    struct { double **linhas; ea551_dtype dtype; } linhas;   //Representa uma matriz completa
}


%token VIRGULA COLCHETE_ESQ COLCHETE_DIR
%token <dval> UINT8_LIT INT32_LIT FLOAT_LIT

%type  <valor>  valor
%type  <linha>  valores linha
%type  <linhas> linhas

%%


//Regra gramatical para o formato da matriz
matriz
    : COLCHETE_ESQ linhas COLCHETE_DIR {
        ea551_mat = malloc(sizeof(*ea551_mat));
        ea551_mat->dtype  = $2.dtype;  //Pega o tipo do conteúdoda matriz
        ea551_mat->linhas = $2.linhas; //Pega a quantidade de linhas da matriz
        print_matrix(ea551_mat);       //Garantir que funcionou na depuração
      }
    ;

linhas
    : linha {
        $$.linhas = NULL;
        arrput($$.linhas, $1.valores);
        $$.dtype = $1.dtype;
      }
    | linhas VIRGULA linha {
        //Verifica se a quantidade de elementos da nova linha é igual a quantidade de colunas da matriz
        ptrdiff_t expect = arrlen($1.linhas[0]);
        ptrdiff_t got    = arrlen($3.valores);
        if (got != expect) {
            fprintf(stderr, "linha %d: esperava %td colunas, veio %td\n",
                    yylineno, expect, got);
            YYERROR;
        }
        $$ = $1;
        arrput($$.linhas, $3.valores);  //Insere o novo vetor de linha no array
        $$.dtype = dtype_max($1.dtype, $3.dtype);  //Caso necessário promove o tipo da matriz 
      }
    ;

linha
    : COLCHETE_ESQ valores COLCHETE_DIR { $$ = $2; } //Pega o array já formado da regra valores e forma a linha
    ;

valores
    : valor {
        //Caso seja o primeiro número de uma linha
        $$.valores = NULL;
        arrput($$.valores, $1.valor);
        $$.dtype = $1.dtype;
      }
      //Caso não, reaproveita a estrutura do anterior e adiciona o próximo valor
    | valores VIRGULA valor {
        $$ = $1;
        arrput($$.valores, $3.valor);
        $$.dtype = dtype_max($1.dtype, $3.dtype);  /* promove o tipo da linha */
      }
    ;

//Recebe o token do léx e atribui seu tipo e enumeração
valor
    : UINT8_LIT  { $$.valor = $1; $$.dtype = EA551_DTYPE_UINT8; }
    | INT32_LIT  { $$.valor = $1; $$.dtype = EA551_DTYPE_INT32; }
    | FLOAT_LIT  { $$.valor = $1; $$.dtype = EA551_DTYPE_FLOAT32; }
    ;

%%


void yyerror(const char *s)
{
    fprintf(stderr, "linha %d: %s\n", yylineno, s);
}

//COnversão de ENUM para string
const char *ea551_dtype_str(ea551_dtype d)
{
    switch (d) {
        case EA551_DTYPE_UINT8:   return "uint8";
        case EA551_DTYPE_INT32:   return "int32";
        case EA551_DTYPE_FLOAT32: return "float32";
    }
    return "?";
}

//Converte com base no tipo atribuido o número de bits preciso
size_t ea551_dtype_size(ea551_dtype d)
{
    switch (d) {
        case EA551_DTYPE_UINT8:   return sizeof(uint8_t);
        case EA551_DTYPE_INT32:   return sizeof(int32_t);
        case EA551_DTYPE_FLOAT32: return sizeof(float);
    }
    return 0;
}

// Limpeza e desalocação da memória RAM
void ea551_mat_free(emat *m)
{
    if (!m) return;
    for (ptrdiff_t i = 0; i < arrlen(m->linhas); i++) arrfree(m->linhas[i]);
    arrfree(m->linhas);
    free(m);
}
