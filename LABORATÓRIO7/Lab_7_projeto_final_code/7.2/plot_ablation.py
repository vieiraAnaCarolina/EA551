import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("resultados_ablacao.csv")
df["tempo_s"] = pd.to_numeric(df["tempo_s"], errors="coerce")
df = df.dropna(subset=["tempo_s"])

base_df = df[df["experimento"] == "base"]
if base_df.empty:
    raise ValueError("Nenhuma linha com experimento == 'base' encontrada em resultados_ablacao.csv")

base_mean = base_df["tempo_s"].mean()

experiments = [
    ("produtores", "Número de Produtores (num_threads)", 1, "ablacao_produtores.png"),
    ("consumidores", "Número de Consumidores (num_consumers)", 1, "ablacao_consumidores.png"),
    ("matmul_threads", "Threads no Matmul (matmul_threads)", 1, "ablacao_matmul_threads.png"),
    ("buffer_size", "Tamanho do Buffer (buffer_size)", 256, "ablacao_buffer_size.png"),
]

for exp_name, xlabel, base_val, out_file in experiments:
    sub = df[df["experimento"] == exp_name].copy()
    sub["valor_num"] = pd.to_numeric(sub["valor"], errors="coerce")
    sub = sub.dropna(subset=["valor_num"])

    stats = (
        sub.groupby("valor_num")["tempo_s"]
        .agg(["mean"])
        .reset_index()
        .sort_values("valor_num")
    )

    if base_val not in stats["valor_num"].values:
        base_row = pd.DataFrame({
            "valor_num": [base_val],
            "mean": [base_mean],
        })
        stats = pd.concat([base_row, stats], ignore_index=True).sort_values("valor_num")

    plt.figure(figsize=(8, 4.8))
    plt.plot(
        stats["valor_num"],
        stats["mean"],
        marker="o",
        linewidth=2,
        label="Média"
    )

    base_y = stats.loc[stats["valor_num"] == base_val, "mean"].iloc[0]
    plt.scatter([base_val], [base_y], color="crimson", zorder=5, s=80, label=f"Baseline ({base_val})")
    plt.annotate(
        f"Base ({base_val}): {base_y:.3f}s",
        (base_val, base_y),
        textcoords="offset points",
        xytext=(8, 8),
        fontsize=9,
        color="crimson"
    )

    if exp_name == "buffer_size":
        plt.ylim(1.0, 2.2)

    plt.xticks(stats["valor_num"], [str(int(v)) for v in stats["valor_num"]])
    plt.title(f"Ablação — {xlabel}")
    plt.xlabel(xlabel)
    plt.ylabel("Tempo de execução no modo teste (s)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Gerado: {out_file}")
