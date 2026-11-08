import sys
import struct
import os
import matplotlib.pyplot as plt
import numpy as np

def display_emat(filepath):
    if not os.path.exists(filepath):
        print(f"[display_emat] Arquivo {filepath} não encontrado.")
        return

    HEADER_SIZE = struct.calcsize("=4sBII")
    with open(filepath, "rb") as f:
        header_bytes = f.read(HEADER_SIZE)
        if len(header_bytes) < HEADER_SIZE:
            return
        magic, dtype, rows, cols = struct.unpack("=4sBII", header_bytes)
        num_pixels = rows * cols
        
        if dtype == 0:
            raw_data = f.read(num_pixels)
            pixels = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32)
        elif dtype == 2:
            raw_data = f.read(num_pixels * 4)
            pixels = np.frombuffer(raw_data, dtype=np.float32)
        else:
            return

    img = pixels.reshape((rows, cols))
    plt.imshow(img, cmap="gray")
    plt.title(f"Amostra: {os.path.basename(filepath)}")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        display_emat(sys.argv[1])