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

void free_node(Node *node) {
    if (!node) return;
    if (node->type == TYPE_STRING) {
        free(node->data.s_val);
    } else if (node->type == TYPE_LIST) {
        int len = arrlen(node->data.list_val);
        for (int i = 0; i < len; i++) free_node(node->data.list_val[i]);
        arrfree(node->data.list_val);
    }
    free(node);
}

Node* get_config_val(const char *key) {
    int total = shlen(tabela);
    for (int i = 0; i < total; i++) {
        if (strcmp(tabela[i].key, key) == 0) return tabela[i].value;
    }
    return NULL;
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
linhas : linhas linha | linha ;
linha : CHAVE ':' item { shput(tabela, $1, $3); } ;
item : VALOR_INT { $$ = $1; } | VALOR_FLOAT { $$ = $1; } | VALOR_STRING { $$ = $1; } | lista { $$ = $1; } ;
lista : '[' ']' { $$ = create_list_node(NULL); } | '[' lista_itens ']' { $$ = create_list_node($2); } ;
lista_itens : lista_itens ',' item { $$ = $1; arrput($$, $3); } | item { $$ = NULL; arrput($$, $1); } ;

%%

int yyerror(const char *s) {
    fprintf(stderr, "Erro de sintaxe YAML: %s\n", s);
    return 0;
}