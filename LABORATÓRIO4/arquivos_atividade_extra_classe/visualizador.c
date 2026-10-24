#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <semaphore.h>

// Define a implementação da fenster para compilação direta
#ifndef FENSTER_C
#define FENSTER_C
#endif
#include "fenster.h" // Inclui a biblioteca fenster

#define SEM_NAME "/sem_emat_pronto"
#define ESCALA 10 // Fator de escala para a janela não ficar minúscula (28x28 -> 280x280)

// Estrutura do cabeçalho de 13 bytes do .emat
typedef struct __attribute__((packed)) {
    char magic[4];     // EMAT
    uint8_t dtype;     // Tipo de dado (0=UINT8)
    uint32_t rows;     // 28
    uint32_t cols;     // 28
} EmatHeader;

// Estrutura memória compartilhada
typedef struct {
    uint32_t rows;
    uint32_t cols;
    uint8_t pixels[28 * 28]; // Buffer para os pixels do MNIST
} SharedData;

// Variáveis globais para gerenciar recursos e sinais
int shm_id = -1;               //id segmento de memória compartilhada
SharedData *shm_ptr = NULL;    //ponteiro para o endereço de memória virtual onde o segmento é compartilhado
sem_t *sem_pronto = NULL;      //Ponteiro para o semáforo 
pid_t child_pid = -1;          //GUarda o PID do filho retornado pelo fork

// Função auxiliar com laço while para garantir a leitura completa de bytes via read
static ssize_t read_all(int fd, void *buf, size_t count) {
    size_t total_lido = 0;
    char *p = (char *)buf;
    while (total_lido < count) {
        ssize_t n = read(fd, p + total_lido, count - total_lido);
        if (n < 0) {
            if (errno == EINTR) continue; // Tenta novamente se for interrompido por sinal
            return -1;                   // Erro real de leitura
        }
        if (n == 0) break; // Fim de arquivo (EOF) prematuro
        total_lido += n;
    }
    return total_lido;
}

// Liberação memória compartilhada e semáforo
void cleanup(void) {
    if (shm_ptr && shm_ptr != (void *)-1) {
        shmdt(shm_ptr);
    }
    if (shm_id != -1) {
        shmctl(shm_id, IPC_RMID, NULL);
    }
    if (sem_pronto && sem_pronto != SEM_FAILED) {
        sem_close(sem_pronto);
        sem_unlink(SEM_NAME);
    }
}

// Handler Ctrl+C 
void sigint_handler(int sig) {
    (void)sig;
    if (child_pid > 0) {
        // Envia SIGTERM para o filho 
        kill(child_pid, SIGTERM);
        waitpid(child_pid, NULL, 0); // Evita processo zumbi
    }
    cleanup();
    printf("\n[PAI] Encerramento por Ctrl+C concluido.\n");
    exit(0);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Uso: %s <caminho_do_arquivo.emat>\n", argv[0]);
        return 1;
    }

    // Configura o manipulador de sinal para SIGINT (Ctrl+C)
    signal(SIGINT, sigint_handler);

    // 1. Criar o semáforo para sincronização
    sem_unlink(SEM_NAME); // Garante que não haja semáforo antigo preso
    sem_pronto = sem_open(SEM_NAME, O_CREAT | O_EXCL, 0666, 0);
    if (sem_pronto == SEM_FAILED) {
        perror("Erro ao criar semáforo sem_open");
        return 1;
    }

    // 2. Criar a área de memória compartilhada
    shm_id = shmget(IPC_PRIVATE, sizeof(SharedData), IPC_CREAT | 0666);
    if (shm_id < 0) {
        perror("Erro ao criar memória compartilhada shmget");
        cleanup();
        return 1;
    }

    // Anexa a memória compartilhada no processo pai
    shm_ptr = (SharedData *)shmat(shm_id, NULL, 0);
    if (shm_ptr == (void *)-1) {
        perror("Erro ao anexar memória compartilhada shmat");
        cleanup();
        return 1;
    }

    // 3. Criar o processo filho com fork
    child_pid = fork();

    if (child_pid < 0) {
        perror("Erro no fork");
        cleanup();
        return 1;
    }

    if (child_pid == 0) { 

        // PROCESSO FILHO (VISUALIZADOR)
        
        // Aguarda a liberação do semáforo pelo pai (evita condição de corrida)
        sem_wait(sem_pronto);

        uint32_t orig_w = shm_ptr->cols;
        uint32_t orig_h = shm_ptr->rows;

        // Janela dimensionada com fator de escala para melhor visualização
        uint32_t win_w = orig_w * ESCALA;
        uint32_t win_h = orig_h * ESCALA;

        uint32_t *buf = calloc(win_w * win_h, sizeof(uint32_t));

        // Preenche o buffer escalado convertendo tom de cinza para ARGB (0x00RRGGBB)
        for (uint32_t y = 0; y < orig_h; y++) {
            for (uint32_t x = 0; x < orig_w; x++) {
                uint8_t gray = shm_ptr->pixels[y * orig_w + x];
                uint32_t pixel_argb = ((uint32_t)gray << 16) | ((uint32_t)gray << 8) | (uint32_t)gray;

                for (uint32_t sy = 0; sy < ESCALA; sy++) {
                    for (uint32_t sx = 0; sx < ESCALA; sx++) {
                        uint32_t px = x * ESCALA + sx;
                        uint32_t py = y * ESCALA + sy;
                        buf[py * win_w + px] = pixel_argb;
                    }
                }
            }
        }

        struct fenster f = {
            .title = "Visualizador .emat - Processo Filho",
            .width = win_w,
            .height = win_h,
            .buf = buf
        };

        fenster_open(&f);

        // Loop de evento da janela
        while (fenster_loop(&f) == 0) {
            // Mantém a janela aberta até ser fechada pelo usuário
        }

        fenster_close(&f);
        free(buf);
        shmdt(shm_ptr);
        sem_close(sem_pronto);

        printf("[FILHO] Janela fechada pelo usuário. Encerrando filho.\n");
        exit(0);

    } else { 

        // PROCESSO PAI (LEITOR)
        
        //Caminho do arquivo em modo apenas leitura
        int fd = open(argv[1], O_RDONLY);

        //Senão achar o arquivo 
        if (fd < 0) {
            perror("Erro ao abrir arquivo .emat");
            kill(child_pid, SIGTERM);
            waitpid(child_pid, NULL, 0);
            cleanup();
            return 1;
        }

        //Leitura e processamento do cabeçalho
        EmatHeader header;
        if (read_all(fd, &header, sizeof(EmatHeader)) != (ssize_t)sizeof(EmatHeader)) {
            fprintf(stderr, "Erro ao ler cabeçalho do arquivo .emat\n");
            close(fd);
            kill(child_pid, SIGTERM);
            waitpid(child_pid, NULL, 0);
            cleanup();
            return 1;
        }

        //Copia os valores  extraidos para a área de memória compartilhada entre os processos
        shm_ptr->rows = header.rows;
        shm_ptr->cols = header.cols;

        // Lê a matriz de píxeis do arquivo direto para a memória compartilhada
        size_t total_pixels = header.rows * header.cols;
        if (read_all(fd, shm_ptr->pixels, total_pixels) != (ssize_t)total_pixels) {
            fprintf(stderr, "Erro ao ler pixels do arquivo .emat\n");
            close(fd);
            kill(child_pid, SIGTERM);
            waitpid(child_pid, NULL, 0);
            cleanup();
            return 1;
        }

        close(fd);

        // Incrementa o semáforo para avisar o filho que a memória está pronta
        sem_post(sem_pronto);

        // Aguarda a finalização do processo filho para evitar processo zumbi
        waitpid(child_pid, NULL, 0);

        // Limpeza dos recursos IPC
        cleanup();
        printf("[PAI] Processo filho finalizado. Pai encerrado.\n");
    }

    return 0;
}