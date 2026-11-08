%{
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Habilita a implementação completa das macros e estruturas do stb_ds.h neste arquivo */
#define STB_DS_IMPLEMENTATION
#include "stb_ds.h"

int yylex(void);
int yyerror(const char *s);

/* Define a estrutura de cada item armazenado na Tabela Hash */
typedef struct {
    char *key;   /* Ponteiro para o nome da chave (string) */
    char *value; /* Ponteiro para o valor formatado como texto plano (string) */
} ItemHash;

/* Ponteiro global para a Tabela Hash gerenciada pela stb_ds.h */
ItemHash *tabela = NULL;

/* 
 * Função Auxiliar: Concatena duas strings alocando um novo bloco de memória no Heap.
 * CRÍTICO: Requer o uso de free() na string retornada após o uso para evitar vazamento.
 */
char *concatenar(const char *s1, const char *s2) {
    char *res = malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(res, s1);
    strcat(res, s2);
    return res;
}
%}

/* Define que todos os símbolos semânticos ($$, $1, $2...) carregam ponteiros char* */
%union {
    char *str;
}

/* Declaração de tokens e não-terminais e a associação com o tipo <str> da union */
%token <str> CHAVE VALOR
%type <str> item lista lista_itens

%%

/* Regra Inicial */
arquivo : linhas ;

linhas : linhas linha
       | linha
       ;

/* 
 * Regra de Linha Parâmetro: CHAVE ':' item
 * - $1: String com a chave (vinda do Flex)
 * - $3: String com o valor formatado (vinda de item)
 */
linha : CHAVE ':' item {
          /* Insere o par na Tabela Hash */
          /* NOTA DO PROFESSOR: Os ponteiros originais de $1 e $3 alocados via strdup/malloc */
          /* não são liberados aqui com free(), gerando vazamentos na memória RAM. */
          shput(tabela, $1, $3);
      }
      ;

/* 
 * Regra de Redução de Item:
 * Repassa o ponteiro de texto do valor primitivo ou da lista diretamente para a regra superior ($$).
 * PROBLEMA ARQUITETURAL: Trata inteiros, floats e matrizes todos indistintamente como "char*".
 */
item : VALOR { $$ = $1; }
     | lista { $$ = $1; }
     ;

/* 
 * Regra de Construção de Listas (Colchetes Externos)
 */
lista : '[' ']' { 
          /* Lista vazia: aloca a string literal "[]" no Heap */
          $$ = strdup("[]"); 
      }
      | '[' lista_itens ']' {
          /* Envolve o conteúdo de lista_itens ($2) entre '[' e ']' */
          char *aux = concatenar("[", $2);
          $$ = concatenar(aux, "]");
          free(aux); /* Libera a memória temporária do passo intermediário */
          /* NOTA: A memória do buffer contido em $2 não foi liberada aqui */
      }
      ;

/* 
 * Regra de Agrupamento Recursivo de Itens da Lista (Intercala vírgulas)
 */
lista_itens : lista_itens ',' item {
                /* Passo recursivo: junta os elementos anteriores ($1) com o novo item ($3) */
                char *aux = concatenar($1, ", ");
                $$ = concatenar(aux, $3);
                free(aux); /* Libera o buffer temporário da junção intermediária */
            }
            | item { 
                /* Caso base: primeiro item da lista, apenas repassa o ponteiro */
                $$ = $1; 
            }
            ;

%%

int main() {
    /* Inicializa a Tabela Hash informando o tratamento de chaves via strdup */
    sh_new_strdup(tabela);

    /* Dispara a análise sintática completa */
    if (yyparse() == 0) {
        printf("=== CONTEUDO DA TABELA HASH ===\n\n");
        int total = shlen(tabela);
        
        /* Iteração sobre os elementos armazenados */
        for (int i = 0; i < total; i++) {
            printf("%s: %s\n", tabela[i].key, tabela[i].value);
        }
        printf("\nTotal de chaves: %d\n", total);
        
        /* 
         * NOTA DO PROFESSOR: A execução encerra sem desalocar os nós e strings 
         * gravados na tabela hash (faltou um laço de limpeza com shfree/free).
         */
    }
    return 0;
}

/* Função chamada automaticamente pelo Bison em caso de falha de sintaxe */
int yyerror(const char *s) {
    fprintf(stderr, "Erro sintatico: %s\n", s);
    return 0;
}