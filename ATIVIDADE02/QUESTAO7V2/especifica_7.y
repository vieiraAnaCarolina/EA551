%{
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "especifica_7.h"

#define STB_DS_IMPLEMENTATION
#include "stb_ds.h"

int yylex(void);
int yyerror(const char *s);

ItemHash *tabela = NULL;

/* Implementação das funções de criação de nós */
Node* create_int_node(int val) {
    Node *n = malloc(sizeof(Node));
    n->type = TYPE_INT;
    n->data.i_val = val;
    return n;
}

Node* create_float_node(double val) {
    Node *n = malloc(sizeof(Node));
    n->type = TYPE_FLOAT;
    n->data.f_val = val;
    return n;
}

Node* create_string_node(char *val) {
    Node *n = malloc(sizeof(Node));
    n->type = TYPE_STRING;
    n->data.s_val = strdup(val);
    return n;
}

Node* create_list_node(Node **list) {
    Node *n = malloc(sizeof(Node));
    n->type = TYPE_LIST;
    n->data.list_val = list;
    return n;
}

void print_node(Node *node) {
    if (!node) return;
    switch (node->type) {
        case TYPE_INT:
            printf("%d", node->data.i_val);
            break;
        case TYPE_FLOAT:
            printf("%.2f", node->data.f_val);
            break;
        case TYPE_STRING:
            printf("\"%s\"", node->data.s_val);
            break;
        case TYPE_LIST: {
            printf("[");
            int len = arrlen(node->data.list_val);
            for (int i = 0; i < len; i++) {
                print_node(node->data.list_val[i]);
                if (i < len - 1) printf(", ");
            }
            printf("]");
            break;
        }
    }
}

void free_node(Node *node) {
    if (!node) return;
    if (node->type == TYPE_STRING) {
        free(node->data.s_val);
    } else if (node->type == TYPE_LIST) {
        int len = arrlen(node->data.list_val);
        for (int i = 0; i < len; i++) {
            free_node(node->data.list_val[i]);
        }
        arrfree(node->data.list_val);
    }
    free(node);
}
%}

%union {
    char *str;
    Node *node;
    Node **list;
}

%token <str> CHAVE
%token <node> VALOR_INT VALOR_FLOAT VALOR_STRING
%type <node> item lista
%type <list> lista_itens

%%

arquivo : linhas ;

linhas : linhas linha
       | linha
       ;

linha : CHAVE ':' item {
          shput(tabela, $1, $3);
      }
      ;

item : VALOR_INT    { $$ = $1; }
     | VALOR_FLOAT  { $$ = $1; }
     | VALOR_STRING { $$ = $1; }
     | lista        { $$ = $1; }
     ;

lista : '[' ']' { 
          $$ = create_list_node(NULL); 
      }
      | '[' lista_itens ']' { 
          $$ = create_list_node($2); 
      }
      ;

lista_itens : lista_itens ',' item {
                $$ = $1;
                arrput($$, $3); /* Adiciona dinamicamente usando a macro do stb_ds.h */
            }
            | item {
                $$ = NULL;
                arrput($$, $1); /* Inicializa o array dinâmico com o primeiro nó */
            }
            ;

%%

int main() {
    sh_new_strdup(tabela);

    if (yyparse() == 0) {
        printf("=== CONTEUDO DA TABELA HASH ===\n\n");
        int total = shlen(tabela);
        for (int i = 0; i < total; i++) {
            printf("%s: ", tabela[i].key);
            print_node(tabela[i].value);
            printf("\n");
        }
        printf("\nTotal de chaves: %d\n", total);

        /* Desalocação limpa de memória para evitar leaks */
        for (int i = 0; i < total; i++) {
            free_node(tabela[i].value);
        }
        shfree(tabela);
    }
    return 0;
}

int yyerror(const char *s) {
    fprintf(stderr, "Erro sintatico: %s\n", s);
    return 0;
}