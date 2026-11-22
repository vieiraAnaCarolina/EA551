#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void sha256(const uint8_t *msg, uint64_t len, uint8_t out[32]);

/* Digest SHA-256 da API_KEY (chave em texto claro nao fica no binario) */
static const char ESPERADO[] = "67c3a3e0f612936a7b0a876121330f42ab531a54ef8bc7af547e53dfbc42548e";

void verificar_api_key(void) {
    const char *key = getenv("API_KEY");
    uint8_t out[32];
    char hex[65];

    if (key == NULL || key[0] == '\0') {
        fprintf(stderr, "[Erro] API_KEY ausente. Acesso negado pela biblioteca.\n");
        exit(1);
    }

    sha256((const uint8_t *)key, (uint64_t)strlen(key), out);
    for (int i = 0; i < 32; i++) {
        sprintf(hex + i * 2, "%02x", out[i]);
    }
    hex[64] = '\0';

    if (strcmp(hex, ESPERADO) != 0) {
        fprintf(stderr, "[Erro] API_KEY invalida (recusada).\n");
        exit(1);
    }
}
