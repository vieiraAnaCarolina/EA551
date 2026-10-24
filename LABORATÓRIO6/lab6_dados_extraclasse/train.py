import os
import time
import argparse
import struct
import posix_ipc
from multiprocessing import shared_memory

import tqdm
import torch
import numpy as np

# Nomes dos semáforos e memória compartilhada POSIX
SHM_NAME = "/ea551_mnist_shm"
SEM_EMPTY_NAME = "/ea551_sem_empty"
SEM_FULL_NAME = "/ea551_sem_full"
SEM_MUTEX_NAME = "/ea551_sem_mutex"

# Tamanhos dos blocos binários (.emat)
BUFFER_SIZE = 256
IMAGE_SIZE = 784
SAMPLE_SIZE = (IMAGE_SIZE * 4) + 4 # 784 floats + 1 int (4 bytes cada)

def train(args):
    print("Conectando à memória compartilhada e semáforos...")
    #Conexão com os semáforos e segmento de memória alocado pelo C
    sem_empty = posix_ipc.Semaphore(SEM_EMPTY_NAME) #Quantas posições do buffer estão livres para o produtor escrever
    sem_full = posix_ipc.Semaphore(SEM_FULL_NAME)   #Quantas posições do buffer contêm dados prontos para leitura
    sem_mutex = posix_ipc.Semaphore(SEM_MUTEX_NAME) #Garante exclusão mútua ao manipular os índices do buffer

    #Memória compartilhada
    shm = shared_memory.SharedMemory(name=SHM_NAME.lstrip('/')) 

    print("Inicialização do modelo de rede neural...")
    model = torch.nn.Linear(28 * 28, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=args["learning_rate"])
    loss_function = torch.nn.CrossEntropyLoss()

    predictions = []
    targets = []
    t_read = 0.0
    t_loop = 0.0

    read_idx = 0
    num_samples = args["num_samples"]

    iterator = tqdm.tqdm(range(num_samples), desc="Consumidor C-SHM lendo dados")

    for _ in iterator:
        t0 = time.perf_counter()

        t_read_start = time.perf_counter()

        #Protocolo de Sincronização POSIX
        sem_full.acquire() 
        sem_mutex.acquire()

        #Cálculo do deslocamento no ring buffer
        sample_offset = read_idx * SAMPLE_SIZE
        
        #Leitura binária direta em bytes sem lock de ponteiro
        img_bytes = bytes(shm.buf[sample_offset : sample_offset + (IMAGE_SIZE * 4)])
        label_bytes = bytes(shm.buf[sample_offset + (IMAGE_SIZE * 4) : sample_offset + SAMPLE_SIZE])

        #Conversão otimizada via NumPy/PyTorch e struct.unpack
        image_np = np.frombuffer(img_bytes, dtype=np.float32).reshape(1, 1, 28, 28)
        image_tensor = torch.from_numpy(image_np)
        image_class = struct.unpack("i", label_bytes)[0]

        #Atualização do ponteiro circular e liberação das travas
        read_idx = (read_idx + 1) % BUFFER_SIZE

        sem_mutex.release()
        sem_empty.release()

        t_read += time.perf_counter() - t_read_start

        # --- ML Loop ---
        image_tensor = image_tensor.view(1, -1)
        output = model(image_tensor)
        loss = loss_function(output, torch.tensor([image_class]))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        predictions.append(output.argmax(dim=1).item())
        targets.append(image_class)

        t_loop += time.perf_counter() - t0
        read_pct = 100.0 * t_read / t_loop if t_loop > 0 else 0.0
        iterator.set_postfix(read_pct=f"{read_pct:.1f}%")

    accuracy = (torch.tensor(predictions) == torch.tensor(targets)).float().mean().item()
    print(f"\nAcurácia de classificação de dígitos: {accuracy:.4f}")
    print(f"O consumo de imagens gastou ({read_pct:.1f}% do tempo do loop)")

    shm.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_samples", type=int, default=60000)
    args = parser.parse_args()
    train(vars(args))