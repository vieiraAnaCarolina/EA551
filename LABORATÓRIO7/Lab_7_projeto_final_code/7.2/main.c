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
    int num_consumers,
    int matmul_threads
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

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    rewind(f);

    long expected_weights_only = (long)(sizeof(float) * NUM_CLASSES * IMAGE_SIZE);
    long expected_with_bias    = expected_weights_only + (long)(sizeof(float) * NUM_CLASSES);

    *weights_out = (float *)malloc(sizeof(float) * NUM_CLASSES * IMAGE_SIZE);
    *bias_out    = NULL;

    if (!*weights_out) {
        perror("Erro ao alocar pesos");
        fclose(f);
        return -1;
    }

    size_t r1 = fread(*weights_out, sizeof(float), NUM_CLASSES * IMAGE_SIZE, f);
    if (r1 != (size_t)(NUM_CLASSES * IMAGE_SIZE)) {
        fprintf(stderr, "Erro: falha ao ler pesos em '%s'\n", model_path);
        free(*weights_out);
        fclose(f);
        return -1;
    }

    if (file_size == expected_with_bias) {
        *bias_out = (float *)malloc(sizeof(float) * NUM_CLASSES);
        if (!*bias_out) {
            perror("Erro ao alocar bias");
            free(*weights_out);
            fclose(f);
            return -1;
        }
        size_t r2 = fread(*bias_out, sizeof(float), NUM_CLASSES, f);
        if (r2 != (size_t)NUM_CLASSES) {
            fprintf(stderr, "Erro: falha ao ler bias em '%s'\n", model_path);
            free(*bias_out);
            free(*weights_out);
            fclose(f);
            return -1;
        }
    } else if (file_size != expected_weights_only) {
        fprintf(stderr, "Aviso: tamanho inesperado (%ld bytes) em '%s'; usando sem bias.\n", file_size, model_path);
    }

    fclose(f);
    return 0;
}

static int run_direct_test_mode(int argc, char **argv) {
    const char *model_path   = "model.bin";
    const char *test_dir     = "../../big_mnist_emat/test";
    int num_threads          = 4;
    int num_samples          = 10000;
    int buffer_size          = 256;
    int num_consumers        = 4;
    int matmul_threads       = 8;

    // Argumentos opcionais apos "direct" ou "--direct":
    // [model_path] [test_dir] [num_threads] [num_consumers] [matmul_threads] [num_samples] [buffer_size]
    if (argc > 2) model_path     = argv[2];
    if (argc > 3) test_dir       = argv[3];
    if (argc > 4) num_threads    = atoi(argv[4]);
    if (argc > 5) num_consumers  = atoi(argv[5]);
    if (argc > 6) matmul_threads = atoi(argv[6]);
    if (argc > 7) num_samples    = atoi(argv[7]);
    if (argc > 8) buffer_size    = atoi(argv[8]);

    if (num_threads <= 0) num_threads = 1;
    if (num_consumers <= 0) num_consumers = 1;
    if (matmul_threads <= 0) matmul_threads = 1;
    if (num_samples <= 0) num_samples = 10000;
    if (buffer_size <= 0) buffer_size = 256;

    printf("======================================================\n");
    printf("[EA551-DIRECT] Modo: test (chamada direta) | Produtores: %d | Consumidores: %d | MatmulThreads: %d | Buffer: %d\n",
           num_threads, num_consumers, matmul_threads, buffer_size);
    printf("[EA551-DIRECT] Modelo: %s | Entrada: %s | Amostras: %d\n",
           model_path, test_dir, num_samples);
    printf("======================================================\n");

    float *weights = NULL, *bias = NULL;
    if (load_model(model_path, &weights, &bias) != 0) {
        return 1;
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    float acc = run_test_from_dataloader(
        weights,
        bias,
        test_dir,
        num_threads,
        num_samples,
        buffer_size,
        num_consumers,
        matmul_threads
    );

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("[Chamada Direta] Acurácia: %.2f%% | [Benchmark C] Tempo total no modo test: %.6f s\n", acc, elapsed);

    free(weights);
    free(bias);
    return 0;
}

int main(int argc, char **argv) {
    if (argc > 1 && (strcmp(argv[1], "direct") == 0 || strcmp(argv[1], "--direct") == 0)) {
        return run_direct_test_mode(argc, argv);
    }

    if (argc > 1) {
        yyin = fopen(argv[1], "r");
        if (!yyin) {
            perror("Erro ao abrir YAML");
            return 1;
        }
    }

    sh_new_strdup(tabela);

    if (yyparse() == 0) {
        Node *n_mode           = get_config_val("mode");
        Node *n_model          = get_config_val("model_path");
        Node *n_threads        = get_config_val("num_threads");
        Node *n_consumers      = get_config_val("num_consumers");
        Node *n_matmul_threads = get_config_val("matmul_threads");
        Node *n_input_path     = get_config_val("input_path");
        Node *n_x11            = get_config_val("enable_x11");
        Node *n_samples        = get_config_val("num_samples");
        Node *n_buf_size       = get_config_val("buffer_size");

        char *mode           = (n_mode && n_mode->type == TYPE_STRING) ? n_mode->data.s_val : "test";
        char *model_path     = (n_model && n_model->type == TYPE_STRING) ? n_model->data.s_val : "model.bin";
        int num_threads      = (n_threads && n_threads->type == TYPE_INT) ? n_threads->data.i_val : 4;
        int num_consumers    = (n_consumers && n_consumers->type == TYPE_INT) ? n_consumers->data.i_val : 4;
        int matmul_threads   = (n_matmul_threads && n_matmul_threads->type == TYPE_INT) ? n_matmul_threads->data.i_val : 1;
        char *input_path     = (n_input_path && n_input_path->type == TYPE_STRING) ? n_input_path->data.s_val : "../../big_mnist_emat/test";
        int enable_x11       = (n_x11 && n_x11->type == TYPE_INT) ? n_x11->data.i_val : 1;
        int num_samples      = (n_samples && n_samples->type == TYPE_INT) ? n_samples->data.i_val : 10000;
        int buffer_size      = (n_buf_size && n_buf_size->type == TYPE_INT) ? n_buf_size->data.i_val : 256;

        if (num_threads <= 0) num_threads = 1;
        if (num_consumers <= 0) num_consumers = 1;
        if (matmul_threads <= 0) matmul_threads = 1;

        printf("======================================================\n");
        printf("[EA551-YAML] Modo: %s | Produtores: %d | Consumidores: %d | MatmulThreads: %d | Buffer: %d | X11: %d\n",
               mode, num_threads, num_consumers, matmul_threads, buffer_size, enable_x11);
        printf("[EA551-YAML] Modelo: %s | Entrada: %s\n", model_path, input_path);
        printf("======================================================\n");

        float *weights = NULL, *bias = NULL;
        if (load_model(model_path, &weights, &bias) != 0) {
            return 1;
        }

        if (strcmp(mode, "test") == 0) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);

            run_test_from_dataloader(
                weights, bias, input_path,
                num_threads, num_samples, buffer_size, num_consumers, matmul_threads
            );

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
