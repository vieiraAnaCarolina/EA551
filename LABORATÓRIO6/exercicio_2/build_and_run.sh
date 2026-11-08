#!/bin/bash
set -e # Aborta o script imediatamente se algum comando falhar

echo "[BUILD] Gerando código C a partir do Bison e Flex..."
bison -d especifica_7.y
flex especifica_7.l

echo "[BUILD] Compilando test_runner e vinculando à libmnist.so..."
gcc -Wall -o test_runner especifica_7.tab.c lex.yy.c -I../exercicio_1 -L../exercicio_1 -lmnist -Wl,-rpath,../exercicio_1 -lm

echo "[EXEC] Executando o módulo testador com config.yaml..."
echo "--------------------------------------------------------"
./test_runner < config.yaml