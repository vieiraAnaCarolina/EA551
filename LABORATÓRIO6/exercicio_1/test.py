#!/usr/bin/env python3
"""
Baseline lento do Modo Teste (referência de funcionamento, não de performance)

- Lê cada imagem de teste via leitura lenta de texto (.txt)
- Multiplica pesos x imagem com loops Python puros
- Reporta acurácia e tempos para referência
"""
import argparse
import os
import time
from typing import List, Dict

import numpy as np
import tqdm
import torch


# =========================================================================
# FUNÇÕES AUXILIARES DE LEITURA LENTA (.txt)
# =========================================================================
def read_mnist_txt_image(filepath: str) -> torch.Tensor:
    """Lê uma imagem .txt tratando caracteres como colchetes e vírgulas."""
    with open(filepath, "r") as f:
        lines = f.readlines()
    
    pixels = []
    for line in lines:
        # Remove colchetes, vírgulas e quebras de linha
        cleaned_line = line.replace('[', '').replace(']', '').replace(',', ' ').strip()
        parts = cleaned_line.split()
        if parts:
            pixels.extend([float(p) for p in parts])
            
    tensor_img = torch.tensor(pixels, dtype=torch.float32).view(1, 1, 28, 28)
    return tensor_img

def get_dict_of_images(root_dir: str) -> Dict[str, int]:
    """Mapeia os caminhos das imagens .txt para as suas respetivas classes (0 a 9)."""
    images_dict = {}
    for digit in range(10):
        digit_dir = os.path.join(root_dir, str(digit))
        if not os.path.exists(digit_dir):
            continue
        for fname in sorted(os.listdir(digit_dir)):
            if fname.endswith(".txt"):
                fpath = os.path.join(digit_dir, fname)
                images_dict[fpath] = digit
    return images_dict


# =========================================================================
# OPERAÇÕES DE TESTE EM PYTHON PURO (MATMUL INGÊNUO)
# =========================================================================
def matmul_ingenuo(W: List[List[float]], x: List[float]) -> List[float]:
    """y = W @ x com três laços explícitos (W: 10×784, x: 784)."""
    nout = len(W)
    nin = len(x)
    y = [0.0] * nout
    for j in range(nout):
        s = 0.0
        row = W[j]
        for i in range(nin):
            s += row[i] * x[i]
        y[j] = s
    return y

def argmax(v: List[float]) -> int:
    best_i = 0
    best = v[0]
    for i in range(1, len(v)):
        if v[i] > best:
            best = v[i]
            best_i = i
    return best_i

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Teste lento do classificador (Python puro + read_txt)."
    )
    parser.add_argument("--mnist_txt_path", type=str, default="mnist_txt")
    parser.add_argument("--weights", type=str, default="W.npy")
    parser.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="Se >0, limita quantas imagens processar.",
    )
    args = parser.parse_args()

    test_root = os.path.join(args.mnist_txt_path, "test")

    if not os.path.exists(args.weights):
        print(f"Erro: Ficheiro de pesos '{args.weights}' não encontrado!")
        return

    print(f"A carregar pesos .npy {args.weights}...")
    W_np = np.load(args.weights)
    W = W_np.astype(np.float64).tolist()

    print(f"A indexar imagens de teste em '{test_root}'...")
    images = get_dict_of_images(test_root)
    items = list(images.items())
    if args.max_images > 0:
        items = items[: args.max_images]
        print(f"(limitado a {len(items)} imagens)")

    correct = 0
    t_read = 0.0
    t_matmul = 0.0
    t0_all = time.perf_counter()

    iterator = tqdm.tqdm(items, desc="test.py (lento)")
    for path, label in iterator:
        t_r0 = time.perf_counter()
        img = read_mnist_txt_image(path)
        t_read += time.perf_counter() - t_r0

        flat = img.view(-1).detach().cpu().tolist()

        t_m0 = time.perf_counter()
        scores = matmul_ingenuo(W, flat)
        pred = argmax(scores)
        t_matmul += time.perf_counter() - t_m0

        if pred == label:
            correct += 1

    elapsed = time.perf_counter() - t0_all
    n = len(items)
    acc = correct / n if n else 0.0
    print(f"\nimagens={n} acertos={correct} acuracia={acc:.4f}")
    print(f"tempo_total={elapsed:.3f}s  leitura={t_read:.3f}s  matmul={t_matmul:.3f}s")

if __name__ == "__main__":
    main()