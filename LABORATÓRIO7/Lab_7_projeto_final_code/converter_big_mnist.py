import os
import struct
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

SRC_DIR = "../big_mnist_txt"
DST_DIR = "../big_mnist_emat"

def converter_arquivo(rel_path):
    src_path = os.path.join(SRC_DIR, rel_path)
    dst_path = os.path.join(DST_DIR, rel_path[:-4] + ".emat")
    
    with open(src_path, "r") as f:
        texto = f.read().replace("[", " ").replace("]", " ").replace(",", " ")
    
    dados = np.fromstring(texto, dtype=np.uint8, sep=" ")
    rows, cols = 128, 128
    
    # Cabeçalho de 13 bytes: magic="EMAT", dtype=0 (UINT8), rows=128, cols=128
    header = struct.pack("<4sBII", b"EMAT", 0, rows, cols)
    
    with open(dst_path, "wb") as f:
        f.write(header)
        f.write(dados.tobytes())

if __name__ == "__main__":
    arquivos = []
    for root, dirs, files in os.walk(SRC_DIR):
        rel_root = os.path.relpath(root, SRC_DIR)
        os.makedirs(os.path.join(DST_DIR, rel_root), exist_ok=True)
        for file in files:
            if file.endswith(".txt"):
                arquivos.append(os.path.join(rel_root, file))

    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(converter_arquivo, arquivos, chunksize=100), total=len(arquivos)))
    print("Conversão concluída!")