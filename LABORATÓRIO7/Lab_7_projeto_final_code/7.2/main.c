#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "stb_ds.h"
#include "especifica_7.h"
#include "shm_common.h"

#ifndef IMAGE_SIZE
#define IMAGE_SIZE 16384
#endif
#ifndef NUM_CLASSES
#define NUM_CLASSES 10
#endif

extern FILE *yyin;
extern int yyparse(void);
extern ItemHash *tabela;
extern Node* get_config_val(const char *key);
extern void free_node(Node *node);

extern float run_test_from_dataloader(
    const float *weights,
    const float *bias,
    const char *test_dir,
    int num_threads,
    int num_samples,
    int buffer_size,
    int num_consumers
);
extern void run_inference_file(
    const char *emat_path,
    const float *weights,
    const float *bias,
    int enable_x11
);

static int load_model(const char *model_path, float **weights_out, float **bias_out) {
    FILE *f = fopen(model_path, "rb");
    if (!f) {
        perror("Erro ao abrir arquivo de pesos model_path");
        return -1;
    }
    *weights_out = (float *)malloc(sizeof(float) * NUM_CLASSES * IMAGE_SIZE);
    *bias_out    = (float *)malloc(sizeof(float) * NUM_CLASSES);

    size_t r1 = fread(*weights_out, sizeof(float), NUM_CLASSES * IMAGE_SIZE, f);
    size_t r2 = fread(*bias_out, sizeof(float), NUM_CLASSES, f);
    fclose(f);

    if (r1 != NUM_CLASSES * IMAGE_SIZE || r2 != NUM_CLASSES) {
        fprintf(stderr, "Erro: tamanho inválido em '%s'\n", model_path);
        free(*weights_out);
        free(*bias_out);
        return -1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc > 1) {
        yyin = fopen(argv[1], "r");
        if (!yyin) {
            perror("Erro ao abrir YAML");
            return 1;
        }
    }

    sh_new_strdup(tabela);

    if (yyparse() == 0) {
        Node *n_mode       = get_config_val("mode");
        Node *n_model      = get_config_val("model_path");
        Node *n_threads    = get_config_val("num_threads");
        Node *n_consumers  = get_config_val("num_consumers");
        Node *n_input_path = get_config_val("input_path");
        Node *n_x11        = get_config_val("enable_x11");
        Node *n_samples    = get_config_val("num_samples");
        Node *n_buf_size   = get_config_val("buffer_size");

        char *mode          = (n_mode && n_mode->type == TYPE_STRING) ? n_mode->data.s_val : "test";
        char *model_path    = (n_model && n_model->type == TYPE_STRING) ? n_model->data.s_val : "model.bin";
        int num_threads     = (n_threads && n_threads->type == TYPE_INT) ? n_threads->data.i_val : 4;
        int num_consumers   = (n_consumers && n_consumers->type == TYPE_INT) ? n_consumers->data.i_val : 4;
        char *input_path    = (n_input_path && n_input_path->type == TYPE_STRING) ? n_input_path->data.s_val : "../../big_mnist_emat/test";
        int enable_x11      = (n_x11 && n_x11->type == TYPE_INT) ? n_x11->data.i_val : 1;
        int num_samples     = (n_samples && n_samples->type == TYPE_INT) ? n_samples->data.i_val : 10000;
        int buffer_size     = (n_buf_size && n_buf_size->type == TYPE_INT) ? n_buf_size->data.i_val : 256;

        if (num_consumers <= 0) num_consumers = 1;

        printf("======================================================\n");
        printf("[EA551-YAML] Modo: %s | Threads: %d | Consumidores: %d | Buffer: %d | X11: %d\n",
               mode, num_threads, num_consumers, buffer_size, enable_x11);
        printf("[EA551-YAML] Modelo: %s | Entrada: %s\n", model_path, input_path);
        printf("======================================================\n");

        float *weights = NULL, *bias = NULL;
        if (load_model(model_path, &weights, &bias) != 0) {
            return 1;
        }

        if (strcmp(mode, "test") == 0) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);

            run_test_from_dataloader(weights, bias, input_path, num_threads, num_samples, buffer_size, num_consumers);

            clock_gettime(CLOCK_MONOTONIC, &end);
            double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
            printf("[Benchmark C] Tempo total no modo test: %.6f s\n", elapsed);
        } else if (strcmp(mode, "infer") == 0) {
            run_inference_file(input_path, weights, bias, enable_x11);
        } else {
            fprintf(stderr, "[Erro] Modo inválido '%s'. Use 'test' ou 'infer'.\n", mode);
        }

        free(weights);
        free(bias);

        int total = shlen(tabela);
        for (int i = 0; i < total; i++) free_node(tabela[i].value);
        shfree(tabela);
    } else {
        fprintf(stderr, "[Erro] Falha ao interpretar o arquivo YAML.\n");
        return 1;
    }

    if (argc > 1 && yyin) fclose(yyin);
    return 0;
}
