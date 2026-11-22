#ifndef ESPECIFICA_7_H
#define ESPECIFICA_7_H

//define um etiquetador para saber qual o tipo de dado armazenado em um determinado nó na memória
typedef enum {
    TYPE_INT,       //inteiro
    TYPE_FLOAT,     //float
    TYPE_STRING,    //string
    TYPE_LIST       //lista
} DataType;

typedef struct Node {
    DataType type;     //armazena a etiqueta identificadora de tipo do nó específico
    union {
        //dependendo do tipo do nó vai armazenar para o correspondente
        int i_val;
        double f_val;
        char *s_val;
        struct Node **list_val; //ponteiro para um vetor dinâmico, armazena o endereço de outros nós.
    } data;
} Node;

//tabela hash
typedef struct {
    char *key;
    Node *value; //ponteiro que aponta para o início da estrutura do Node
} ItemHash;

//funções auxiliares 
Node* create_int_node(int val);
Node* create_float_node(double val);
Node* create_string_node(char *val);
Node* create_list_node(Node **list);
void print_node(Node *node);
void free_node(Node *node);

#endif


