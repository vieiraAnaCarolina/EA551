#!/bin/bash

EXEC_LAB4="../../LABORATÓRIO4/arquivos_atividade_extra_classe/matmul"
EXEC_LAB5="./matmul"
TESTES_DIR="testes_4.3"

echo "=========================================================="
echo "    BENCHMARK COMPARATIVO: LAB 4 vs LAB 5 (Otimizado)"
echo "=========================================================="
printf "%-10s | %-15s | %-15s | %-10s\n" "Teste" "Lab 4 (s)" "Lab 5 (s)" "Speedup"
echo "----------------------------------------------------------"

for i in {1..3}; do
    FILE_A="${TESTES_DIR}/A_${i}.emat"
    FILE_B="${TESTES_DIR}/B_${i}.emat"
    FILE_C="${TESTES_DIR}/C_${i}.emat"
    ARGS="$FILE_A $FILE_B $FILE_C"

    if [ -f "$EXEC_LAB4" ]; then
        TIME_LAB4=$({ time $EXEC_LAB4 $ARGS > /dev/null; } 2>&1 | grep real | grep -oE '[0-9]+\.[0-9]+' | tail -n 1)
    else
        TIME_LAB4="N/A"
    fi

    TIME_LAB5=$( $EXEC_LAB5 $ARGS 2>&1 | grep "Tempo de execução" | grep -oE '[0-9]+\.[0-9]+' )

    if [ -z "$TIME_LAB5" ]; then TIME_LAB5="N/A"; fi

    if [ "$TIME_LAB4" != "N/A" ] && [ "$TIME_LAB5" != "N/A" ]; then
        SPEEDUP=$(echo "scale=2; $TIME_LAB4 / $TIME_LAB5" | bc 2>/dev/null)
        SPEEDUP="${SPEEDUP}x"
    else
        SPEEDUP="N/A"
    fi

    printf "%-10s | %-15s | %-15s | %-10s\n" "Teste $i" "$TIME_LAB4" "$TIME_LAB5" "$SPEEDUP"
done

echo "=========================================================="