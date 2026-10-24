#!/bin/bash

# Define as pastas de origem (.txt) e destino (.emat)
ORIGEM="mnist_txt"
DESTINO="mnist_emat"

# Verifica se o executável compilado do C está presente na pasta
if [ ! -f ./ea551-mat ]; then
    echo "Erro: Executável ./ea551-mat não encontrado nesta pasta!"
    exit 1
fi

echo "Iniciando a conversão de $ORIGEM para $DESTINO..."

# Busca todos os arquivos .txt e processa um por um preservando a estrutura
find "$ORIGEM" -type f -name "*.txt" | while read -r txt_file; do
    emat_file="${txt_file/$ORIGEM/$DESTINO}"
    emat_file="${emat_file%.txt}.emat"

    dir_destino=$(dirname "$emat_file")
    mkdir -p "$dir_destino"

    ./ea551-mat < "$txt_file" > "$emat_file"
done

echo "Conversão concluída com sucesso! Arquivos salvos em '$DESTINO'."
