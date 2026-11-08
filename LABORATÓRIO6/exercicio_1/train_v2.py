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

# =========================================================================
# 1. MAPEAMENTO DE CONSTANTES DE MEMÓRIA COMPARTILHADA (shm_common.h)
# =========================================================================
SHM_NAME = "/ea551_mnist_shm"
SEM_EMPTY_NAME = "/ea551_sem_empty"
SEM_FULL_NAME = "/ea551_sem_full"
SEM_MUTEX_NAME = "/ea551_sem_mutex"

BUFFER_SIZE = 256
IMAGE_SIZE = 784
SAMPLE_SIZE = (IMAGE_SIZE * 4) + 4  # 784 floats (32-bit) + 1 int (32-bit)

# =========================================================================
# 2. CARREGAMENTO E MAPEAMENTO DA BIBLIOTECA COMPARTILHADA (libmnist.so)
# =========================================================================
LIB_PATH = os.path.abspath("./libmnist.so")

if not os.path.exists(LIB_PATH):
    raise FileNotFoundError(
        f"Biblioteca {LIB_PATH} não encontrada. Compile o .so primeiro."
    )

libmnist = ctypes.CDLL(LIB_PATH)

# void start_dataloader(int num_threads, const char *emat_dir)
libmnist.start_dataloader.argtypes = [ctypes.c_int, ctypes.c_char_p]
libmnist.start_dataloader.restype = None

# void stop_dataloader(void)
if hasattr(libmnist, "stop_dataloader"):
    libmnist.stop_dataloader.argtypes = []
    libmnist.stop_dataloader.restype = None

# void run_inference(const float *pixels, const float *weights, const float *bias, const char *image_path)
libmnist.run_inference.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_char_p,
]
libmnist.run_inference.restype = None

# float run_test_dataset(const float *weights, const float *bias, const float *test_images, const int *test_labels, int num_samples)
libmnist.run_test_dataset.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
]
libmnist.run_test_dataset.restype = ctypes.c_float

# int predict_single_sample(const float *pixels, const float *weights, const float *bias)
if hasattr(libmnist, "predict_single_sample"):
    libmnist.predict_single_sample.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # pixels
        ctypes.POINTER(ctypes.c_float),  # weights
        ctypes.POINTER(ctypes.c_float)   # bias
    ]
    libmnist.predict_single_sample.restype = ctypes.c_int


# =========================================================================
# FUNÇÃO PRINCIPAL DE TREINO E INTEGRAÇÃO COM C
# =========================================================================
def train_and_evaluate(args):
    # ---------------------------------------------------------------------
    # ETAPA A: Dispara o DataLoader em C numa THREAD SEPARADA (PRODUTOR - TREINO)
    # ---------------------------------------------------------------------
    print(
        f"\n[DataLoader C] Disparando o produtor nativo em background ({args['threads']} threads) em '{args['train_dir']}'..."
    )

    producer_thread = threading.Thread(
        target=libmnist.start_dataloader,
        args=(args["threads"], args["train_dir"].encode("utf-8")),
    )
    producer_thread.start()

    time.sleep(0.5)  # Pausa para sincronização de alocação de semáforos no C

    print("[Python] Conectando à memória compartilhada POSIX e semáforos...")
    sem_empty = posix_ipc.Semaphore(SEM_EMPTY_NAME)
    sem_full = posix_ipc.Semaphore(SEM_FULL_NAME)
    sem_mutex = posix_ipc.Semaphore(SEM_MUTEX_NAME)

    shm = shared_memory.SharedMemory(name=SHM_NAME.lstrip("/"))

    # ---------------------------------------------------------------------
    # ETAPA B: Inicialização e Treino da Rede Neural PyTorch (CONSUMIDOR) 
    # ---------------------------------------------------------------------
    print(
        "\n[PyTorch] Inicializando o modelo de regressão logística / Linear(784 -> 10)..."
    )
    model = torch.nn.Linear(IMAGE_SIZE, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=args["learning_rate"])
    loss_function = torch.nn.CrossEntropyLoss()

    read_idx = 0
    num_samples = args["num_samples"]

    predictions = []
    targets = []

    t_read = 0.0
    t_loop = 0.0

    iterator = tqdm.tqdm(
        range(num_samples), desc="[Treino] Consumindo amostras da SHM"
    )

    for _ in iterator:
        t0 = time.perf_counter()
        t_read_start = time.perf_counter()

        sem_full.acquire()
        sem_mutex.acquire()

        sample_offset = read_idx * SAMPLE_SIZE
        img_bytes = bytes(
            shm.buf[sample_offset : sample_offset + (IMAGE_SIZE * 4)]
        )
        label_bytes = bytes(
            shm.buf[
                sample_offset + (IMAGE_SIZE * 4) : sample_offset + SAMPLE_SIZE
            ]
        )

        image_np = np.frombuffer(img_bytes, dtype=np.float32)
        image_tensor = torch.from_numpy(image_np).view(1, -1)
        image_class = struct.unpack("i", label_bytes)[0]

        read_idx = (read_idx + 1) % BUFFER_SIZE

        sem_mutex.release()
        sem_empty.release()

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

    train_acc = (
        (torch.tensor(predictions) == torch.tensor(targets))
        .float()
        .mean()
        .item()
        * 100.0
    )
    print(
        f"\n[Treino Concluído] Acurácia no conjunto de treino: {train_acc:.2f}%"
    )

    shm.close()
    try:
        shm.unlink()
    except FileNotFoundError:
        pass

    if hasattr(libmnist, "stop_dataloader"):
        libmnist.stop_dataloader()
    producer_thread.join()

    # ---------------------------------------------------------------------
    # ETAPA C: MODO TESTE ACELERADO EM C (USANDO DATALOADER + SHM + MATMUL)
    # ---------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" INICIANDO MODO TESTE ACELERADO NA LIBMNIST.SO (DATALOADER)")
    print("=" * 60)

    # 1. Extrai os pesos W (10x784) e bias b (10) aprendidos no treino
    weights = model.weight.data.cpu().numpy().astype(np.float32).flatten()
    bias = model.bias.data.cpu().numpy().astype(np.float32).flatten()

    weights_ptr = weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    bias_ptr = bias.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    NUM_TEST_SAMPLES = 10000

    # 2. Inicia o DataLoader em C para o diretório de TESTE
    print(f"[DataLoader C Teste] Carregando imagens de '{args['test_dir']}'...")
    test_producer_thread = threading.Thread(
        target=libmnist.start_dataloader,
        args=(args["threads"], args["test_dir"].encode("utf-8")),
    )
    test_producer_thread.start()

    time.sleep(0.5)

    test_sem_empty = posix_ipc.Semaphore(SEM_EMPTY_NAME)
    test_sem_full = posix_ipc.Semaphore(SEM_FULL_NAME)
    test_sem_mutex = posix_ipc.Semaphore(SEM_MUTEX_NAME)
    test_shm = shared_memory.SharedMemory(name=SHM_NAME.lstrip("/"))

    test_read_idx = 0
    correct_predictions = 0

    # Guardaremos a primeira amostra do teste para usar na demonstração da Etapa D se necessário
    sample_pixels_demo = None
    sample_label_demo = None

    t_start = time.perf_counter()

    for i in range(NUM_TEST_SAMPLES):
        test_sem_full.acquire()
        test_sem_mutex.acquire()

        sample_offset = test_read_idx * SAMPLE_SIZE
        img_bytes = bytes(
            test_shm.buf[sample_offset : sample_offset + (IMAGE_SIZE * 4)]
        )
        label_bytes = bytes(
            test_shm.buf[
                sample_offset + (IMAGE_SIZE * 4) : sample_offset + SAMPLE_SIZE
            ]
        )

        image_np = np.frombuffer(img_bytes, dtype=np.float32)
        image_class = struct.unpack("i", label_bytes)[0]

        test_read_idx = (test_read_idx + 1) % BUFFER_SIZE

        test_sem_mutex.release()
        test_sem_empty.release()

        # Guarda a primeira amostra para o modo de inferência individual
        if i == 0:
            sample_pixels_demo = image_np.copy()
            sample_label_demo = image_class

        # Executa a inferência acelerada via matmul em C
        sample_ptr = image_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        if hasattr(libmnist, "predict_single_sample"):
            pred_label = libmnist.predict_single_sample(sample_ptr, weights_ptr, bias_ptr)
        else:
            # Fallback local se predict_single_sample não estiver no C
            scores = np.dot(weights.reshape(10, 784), image_np) + bias
            pred_label = int(np.argmax(scores))

        if pred_label == image_class:
            correct_predictions += 1

    t_end = time.perf_counter()

    test_shm.close()
    try:
        test_shm.unlink()
    except FileNotFoundError:
        pass

    if hasattr(libmnist, "stop_dataloader"):
        libmnist.stop_dataloader()
    test_producer_thread.join()

    test_accuracy = (correct_predictions / NUM_TEST_SAMPLES) * 100.0
    print(f"[C - DataLoader Test] Acurácia no conjunto de teste ({NUM_TEST_SAMPLES} amostras): {test_accuracy:.2f}%")
    print(f"[BENCHMARK C] Teste com DataLoader concluído em {t_end - t_start:.6f} segundos.")

    # ---------------------------------------------------------------------
    # ETAPA D: MODO INFERÊNCIA INDIVIDUAL EM C (FORK + VISUALIZAÇÃO)
    # ---------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" INICIANDO MODO INFERÊNCIA INDIVIDUAL EM C (FORK + VISUALIZAÇÃO)")
    print("=" * 60)

    image_path = args.get("image")

    if image_path:
        # --- MODO 1: Arquivo especificado via CLI (--image) ---
        if os.path.exists(image_path):
            print(f"\n--- Processando arquivo especificado: {image_path} ---")

            HEADER_SIZE = struct.calcsize("=4sBII")  # 13 bytes
            with open(image_path, "rb") as f:
                header_bytes = f.read(HEADER_SIZE)
                magic, dtype, rows, cols = struct.unpack("=4sBII", header_bytes)
                num_pixels = rows * cols

                if dtype == 0:  # UINT8
                    raw_data = f.read(num_pixels)
                    sample_pixels = (
                        np.frombuffer(raw_data, dtype=np.uint8)
                        .astype(np.float32)
                        / 255.0
                    )
                else:  # FLOAT32
                    raw_data = f.read(num_pixels * 4)
                    sample_pixels = np.frombuffer(raw_data, dtype=np.float32)

            sample_pixels_ptr = sample_pixels.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            sample_path_bytes = image_path.encode("utf-8")

            libmnist.run_inference(sample_pixels_ptr, weights_ptr, bias_ptr, sample_path_bytes)
        else:
            print(
                f"[ERRO] O arquivo especificado '{image_path}' não foi encontrado."
            )

    elif sample_pixels_demo is not None:
        # --- MODO 2: Execução padrão (Amostra do teste consumida via SHM) ---
        sample_path_str = "sample_inferencia_0.emat"

        # Salva a imagem lida da SHM temporariamente no disco para o display_emat.py
        header_bytes = struct.pack("=4sBII", b"EMAT", 2, 28, 28)
        with open(sample_path_str, "wb") as f:
            f.write(header_bytes)
            f.write(sample_pixels_demo.tobytes())

        sample_pixels_ptr = sample_pixels_demo.ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )
        sample_path_bytes = sample_path_str.encode("utf-8")

        print(
            f"\n--- Testando Amostra da SHM (Rótulo real: {sample_label_demo}) ---"
        )
        libmnist.run_inference(
            sample_pixels_ptr, weights_ptr, bias_ptr, sample_path_bytes
        )

# =========================================================================
# PONTO DE ENTRADA
# =========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Treino MNIST com biblioteca C nativa libmnist.so"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Taxa de aprendizado",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=60000,
        help="Quantidade de amostras para treino",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Número de threads do DataLoader em C",
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default="./mnist_emat/train",
        help="Diretório de treino (.emat)",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="./mnist_emat/test",
        help="Diretório de teste (.emat)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Caminho para um arquivo .emat específico para o modo inferência individual",
    )

    args = parser.parse_args()
    train_and_evaluate(vars(args))