#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

//Estrutura .emat
typedef struct {
    uint32_t rows;
    uint32_t cols;
    float *data;
} EMat;

// ./npy2emat W.npy W.emat
int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Uso: %s <arquivo_entrada.npy> <arquivo_saida.emat>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *npy_path = argv[1];  //Caminho .npy
    const char *emat_path = argv[2]; //Caminho .emat será salvo

    //Abrir o arquivo .npy usando chamadas de sistema
    int fd_in = open(npy_path, O_RDONLY);
    if (fd_in < 0) {
        perror("Erro ao abrir arquivo .npy de entrada");
        return EXIT_FAILURE;
    }

    //Estrutura para obter os metadados do arquivo passado
    struct stat st;
    if (fstat(fd_in, &st) < 0) {
        perror("Erro ao obter tamanho do arquivo .npy");
        close(fd_in);
        return EXIT_FAILURE;
    }
    //Inteiro sem sinal
    size_t file_size = st.st_size;

    // Alocar buffer e ler todo o arquivo usando read()
    unsigned char *buffer = malloc(file_size);
    if (!buffer) {
        perror("Erro ao alocar memoria para o buffer");
        close(fd_in);
        return EXIT_FAILURE;
    }

    //Retorna número total de bytes que foram lidos
    ssize_t bytes_read = read(fd_in, buffer, file_size);
    if (bytes_read < 0) {
        perror("Erro na leitura do arquivo com read()");
        free(buffer);
        close(fd_in);
        return EXIT_FAILURE;
    }
    close(fd_in);

    //Comparação mágica inicial (\x93NUMPY)
    if (memcmp(buffer, "\x93NUMPY", 6) != 0) {
        fprintf(stderr, "Erro: Arquivo nao e um formato .npy valido.\n");
        free(buffer);
        return EXIT_FAILURE;
    }

    //Extrair o tamanho do cabeçalho (HEADER_LEN) - bytes 8 e 9 (uint16_t)
    uint16_t header_len = *(uint16_t *)&buffer[8];
    //Onde começam os dados numéricos brutos
    size_t data_offset = 10 + header_len; // 6 (magic) + 1 (major) + 1 (minor) + 2 (header_len) + header_len

    //Parser básico do cabeçalho em ASCII para extrair a 'shape'
    char header[ header_len + 1 ];
    memcpy(header, &buffer[10], header_len);
    header[header_len] = '\0';

    //Procurar a chave shape no dicionario ASCII
    char *shape_ptr = strstr(header, "'shape':");
    if (!shape_ptr) {
        shape_ptr = strstr(header, "\"shape\":");
    }

    uint32_t rows = 0, cols = 0;
    if (shape_ptr) {
        sscanf(strchr(shape_ptr, '('), "(%u, %u)", &rows, &cols);
    }

    if (rows == 0 || cols == 0) {
        fprintf(stderr, "Erro ao extrair as dimensoes (shape) do cabecalho .npy\n");
        free(buffer);
        return EXIT_FAILURE;
    }

    printf("[INFO] Dimensoes detectadas no .npy: %u x %u\n", rows, cols);

    //Escrever o arquivo .emat usando chamadas de sistema (open/write)
    int fd_out = open(emat_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd_out < 0) {
        perror("Erro ao criar arquivo .emat de saída");
        free(buffer);
        return EXIT_FAILURE;
    }

    //Escrever cabeçalho .emat (rows e cols como uint32_t)
    if (write(fd_out, &rows, sizeof(uint32_t)) != sizeof(uint32_t) ||
        write(fd_out, &cols, sizeof(uint32_t)) != sizeof(uint32_t)) {
        perror("Erro ao escrever cabecalho .emat");
        close(fd_out);
        free(buffer);
        return EXIT_FAILURE;
    }

    //Escrever os dados brutos dos pesos
    size_t data_size = rows * cols * sizeof(float);
    if (write(fd_out, &buffer[data_offset], data_size) != (ssize_t)data_size) {
        perror("Erro ao escrever os dados dos pesos no .emat");
        close(fd_out);
        free(buffer);
        return EXIT_FAILURE;
    }

    close(fd_out);
    free(buffer);

    printf("[SUCESSO] Conversao concluida: %s -> %s\n", npy_path, emat_path);
    return EXIT_SUCCESS;
}