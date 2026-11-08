#!/usr/bin/env python3
import argparse
import ctypes
import mmap
import os
import time
import numpy as np
import posix_ipc
import torch
import tqdm

#Imagem do MNIST na RAM
class SharedImage(ctypes.Structure):
    _fields_ = [
        ("pixels", ctypes.c_float * 784), #Imagem de 28x28
        ("label", ctypes.c_int32),        #Rótulo Imagem (0 a 9)
    ]

#Estrutura do buffer compartilhado
class SharedBuffer(ctypes.Structure):
    _fields_ = [
        ("buffer", SharedImage * 256), #256 imagens
        ("write_idx", ctypes.c_int32), #indice de escrita
        ("read_idx", ctypes.c_int32),  #indice de leitura
        ("is_done", ctypes.c_int32),   #flag producao de todas feita
    ]

#Sincronização dos processos, mapeamento de memória e pytorch
def train_c_shm(args):
    print("Conectando aos semáforos e à memória compartilhada POSIX...")
    
    sem_empty = posix_ipc.Semaphore("/ea551_sem_empty")  #Quantas posições vagas restam no buffer
    sem_full = posix_ipc.Semaphore("/ea551_sem_full")    #Quantas posições têm dados prontos para leitura
    sem_mutex = posix_ipc.Semaphore("/ea551_sem_mutex")  #Semáforo


    shm = posix_ipc.SharedMemory("/ea551_mnist_shm")            #abre memoria alocada pelo produtor
    mapped_mem = mmap.mmap(shm.fd, ctypes.sizeof(SharedBuffer)) #mapeia o fd para end do python
    shm_buf = SharedBuffer.from_buffer(mapped_mem)              #python acessar o buffer como um objeto nativo

    print("Inicialização do modelo de rede neural (sem bias e com normalização / 255.0)...")
    model = torch.nn.Linear(28 * 28, 10, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=args["learning_rate"])
    loss_function = torch.nn.CrossEntropyLoss()

    #Para o cáculo da acurácia
    predictions = []
    targets = []
    t_read = 0.0
    t_loop = 0.0

    num_samples = args["num_samples"]
    iterator = tqdm.tqdm(range(num_samples), desc="Consumidor C-SHM lendo dados e treinando")

    for _ in iterator:

        #Marcação tempo
        t0 = time.perf_counter()
        t_read_start = time.perf_counter()

        sem_full.acquire()   #verifica se tem algum dado pronto no buffer
        sem_mutex.acquire()  #garantia nenhum outro processo altere enquanto lê a posição atual

        #Processo de leitura e extração dos dados da RAM
        idx = shm_buf.read_idx           #Lê o ponteiro atual
        img_struct = shm_buf.buffer[idx] #Acessa a sharedimage pelo index

        pixels_np = np.ctypeslib.as_array(img_struct.pixels).copy() #Converte e extrai o rótulo
        image_class = int(img_struct.label) 

        shm_buf.read_idx = (idx + 1) % 256 #Incrementa o ponteiro de leitura da fila 

        sem_mutex.release() #Abre o mutex, c pode acessar buffer
        sem_empty.release() #Incrementa o semáforo de posições livres


        #Tempo de leitura, treinamento rede neural e coleta métricas de desempenho
        image_tensor = (torch.from_numpy(pixels_np) / 255.0).view(1, -1)
        t_read += time.perf_counter() - t_read_start

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
    print(f"\nAcurácia de classificação de dígitos (treino): {accuracy:.4f}")
    print(f"O consumo de imagens gastou ({read_pct:.1f}% do tempo do loop)")

    out_dir = args.get("weights_dir", ".")
    os.makedirs(out_dir, exist_ok=True)
    w_path = os.path.join(out_dir, "W.npy")
    np.save(w_path, model.weight.detach().cpu().numpy().astype(np.float32))
    print(f"Pesos da rede neural salvos em {w_path} (shape {tuple(model.weight.shape)})")

    del shm_buf
    try:
        mapped_mem.close()
    except BufferError:
        pass
    shm.close_fd()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consumidor C-SHM para o train_v2.2.py")
    parser.add_argument("--num_samples", type=int, default=60000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weights_dir", type=str, default=".")
    args = parser.parse_args()

    train_c_shm(vars(args))
    print("Treinamento concluído com sucesso.")
