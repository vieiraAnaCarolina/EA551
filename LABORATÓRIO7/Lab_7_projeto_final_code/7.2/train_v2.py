import argparse
import ctypes
import os
import struct
import threading
import time
from multiprocessing import shared_memory

import numpy as np
import posix_ipc
import torch
import tqdm

SHM_NAME = "/ea551_mnist_shm"
SEM_EMPTY_NAME = "/ea551_sem_empty"
SEM_FULL_NAME = "/ea551_sem_full"
SEM_MUTEX_NAME = "/ea551_sem_mutex"

BUFFER_SIZE = 256
IMAGE_DIM = 128
IMAGE_SIZE = IMAGE_DIM * IMAGE_DIM  # 16384
SAMPLE_SIZE = (IMAGE_SIZE * 4) + 4

LIB_PATH = os.path.abspath("./libmnist.so")
if not os.path.exists(LIB_PATH):
    raise FileNotFoundError(f"Biblioteca {LIB_PATH} não encontrada.")

libmnist = ctypes.CDLL(LIB_PATH)

libmnist.start_dataloader.argtypes = [ctypes.c_int, ctypes.c_char_p]
libmnist.start_dataloader.restype = None

libmnist.stop_dataloader.argtypes = []
libmnist.stop_dataloader.restype = None

libmnist.run_test_from_dataloader.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
libmnist.run_test_from_dataloader.restype = ctypes.c_float


def train_and_evaluate(args):
    print(f"[PID {os.getpid()}] Iniciando treino 128x128 (W: 10x16384, sem bias) com libmnist.so...")

    buf_size = max(1, min(args["buffer_size"], BUFFER_SIZE))

    producer_thread = threading.Thread(
        target=libmnist.start_dataloader,
        args=(args["threads"], args["train_dir"].encode("utf-8")),
    )
    producer_thread.start()
    time.sleep(0.5)

    sem_empty = posix_ipc.Semaphore(SEM_EMPTY_NAME)
    sem_full = posix_ipc.Semaphore(SEM_FULL_NAME)
    sem_mutex = posix_ipc.Semaphore(SEM_MUTEX_NAME)
    shm = shared_memory.SharedMemory(name=SHM_NAME.lstrip("/"))

    # Modificação V3: classificador puramente linear y = Ax (sem bias)
    model = torch.nn.Linear(IMAGE_SIZE, 10, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=args["learning_rate"])
    loss_function = torch.nn.CrossEntropyLoss()

    read_idx = 0
    num_samples = args["num_samples"]
    predictions, targets = [], []

    iterator = tqdm.tqdm(range(num_samples), desc="[Treino 128x128] Consumindo SHM")
    for _ in iterator:
        sem_full.acquire()
        sem_mutex.acquire()

        sample_offset = read_idx * SAMPLE_SIZE
        img_bytes = bytes(shm.buf[sample_offset : sample_offset + (IMAGE_SIZE * 4)])
        label_bytes = bytes(shm.buf[sample_offset + (IMAGE_SIZE * 4) : sample_offset + SAMPLE_SIZE])

        image_np = np.frombuffer(img_bytes, dtype=np.float32).copy()
        image_tensor = torch.from_numpy(image_np).view(1, -1)
        image_class = struct.unpack("i", label_bytes)[0]

        read_idx = (read_idx + 1) % buf_size
        sem_mutex.release()
        sem_empty.release()

        output = model(image_tensor)
        loss = loss_function(output, torch.tensor([image_class]))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        predictions.append(output.argmax(dim=1).item())
        targets.append(image_class)

    train_acc = (torch.tensor(predictions) == torch.tensor(targets)).float().mean().item() * 100.0
    print(f"\n[Treino Concluído] Acurácia no treino: {train_acc:.2f}%")

    shm.close()
    try:
        shm.unlink()
    except FileNotFoundError:
        pass
    libmnist.stop_dataloader()
    producer_thread.join()

    # Salva W (10x16384 float32) e vetor de bias zerado (10 float32) para manter compatibilidade com C
    weights = model.weight.data.cpu().numpy().astype(np.float32).flatten()
    bias = np.zeros(10, dtype=np.float32)

    with open(args["model_out"], "wb") as f:
        f.write(weights.tobytes())
        f.write(bias.tobytes())
    print(f"[Modelo Salvo] Pesos W (10x16384) salvos em '{args['model_out']}' ({os.path.getsize(args['model_out'])} bytes).")

    print("\n[Teste] Executando avaliação no conjunto de teste via libmnist.so...")
    weights_ptr = weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    bias_ptr = bias.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    libmnist.run_test_from_dataloader(
        weights_ptr,
        bias_ptr,
        args["test_dir"].encode("utf-8"),
        args["threads"],
        args["test_samples"],
        buf_size,
        args["consumers"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treino Big MNIST 128x128 com libmnist.so (sem bias)")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_samples", type=int, default=60000)
    parser.add_argument("--test_samples", type=int, default=10000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--consumers", type=int, default=4)
    parser.add_argument("--buffer_size", type=int, default=BUFFER_SIZE)
    parser.add_argument("--train_dir", type=str, default="../../big_mnist_emat/train")
    parser.add_argument("--test_dir", type=str, default="../../big_mnist_emat/test")
    parser.add_argument("--model_out", type=str, default="model.bin")
    args = parser.parse_args()
    train_and_evaluate(vars(args))