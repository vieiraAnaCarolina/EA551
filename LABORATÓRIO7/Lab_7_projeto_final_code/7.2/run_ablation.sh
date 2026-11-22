#!/usr/bin/env bash
set -e

OUT="resultados_ablacao.csv"
echo "experimento,parametro,valor,execucao,tempo_s" > "$OUT"

run_one() {
    local exp="$1"
    local param="$2"
    local val="$3"
    local yaml_file="$4"
    local reps=3

    for r in $(seq 1 $reps); do
        output=$(./ea551_runner "$yaml_file")
        tempo=$(echo "$output" | awk -F'Tempo total no modo test: ' '/Tempo total no modo test:/ {print $2}' | awk '{print $1}')
        echo "${exp},${param},${val},${r},${tempo}" >> "$OUT"
        echo "[${exp} | ${param}=${val} | run ${r}] -> ${tempo} s"
    done
}

run_one "base" "baseline" "1" "ablation_yamls/base.yaml"

for p in 2 4 8 16; do
    run_one "produtores" "num_threads" "$p" "ablation_yamls/prod_${p}.yaml"
done

for c in 2 4 8 16 32 64; do
    run_one "consumidores" "num_consumers" "$c" "ablation_yamls/cons_${c}.yaml"
done

for m in 2 4 8; do
    run_one "matmul_threads" "matmul_threads" "$m" "ablation_yamls/matmul_${m}.yaml"
done

for b in 16 32 64 128 256 512; do
    run_one "buffer_size" "buffer_size" "$b" "ablation_yamls/buf_${b}.yaml"
done

echo "Concluído! Dados salvos em $OUT"
