/* Confere API_KEY pelo digest SHA-256.
 * gcc -o chave chave.c sha256.c
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void sha256(const uint8_t *msg, uint64_t len, uint8_t out[32]);

/* Digest de "ea551". Troque pelo digest da sua senha. */
static const char ESPERADO[] =
    "f38b7d6371e53d2c80dedb04817aeb1e2457aec5b76df5e0803707d96818a7ad";

int main(void) {
    const char *key = getenv("API_KEY");
    uint8_t out[32];
    char hex[65];
    int i;

    if (key == NULL) {
        fprintf(stderr, "API_KEY ausente\n");
        return 1;
    }

    sha256((const uint8_t *)key, (uint64_t)strlen(key), out);
    for (i = 0; i < 32; i++) {
        sprintf(hex + i * 2, "%02x", out[i]);
    }
    hex[64] = '\0';
    printf("%s\n", hex);

    if (strcmp(hex, ESPERADO) != 0) {
        printf("recusada\n");
        return 1;
    }
    printf("aceita\n");
    return 0;
}
