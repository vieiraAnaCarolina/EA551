# `big_mnist_txt` — gerador

`gerar_big_mnist.py` lê um `mnist_txt` 28×28 (pastas `train/<dígito>/` e `test/<dígito>/`) e escreve outro conjunto no mesmo formato de texto, com matrizes 128×128.

Cada pixel da imagem grande é o pixel mais próximo da imagem original (`y * 28 // 128`, `x * 28 // 128`). O rótulo continua sendo o nome da pasta. Não são dígitos novos.

| Split | Entrada | Saída |
|-------|---------|--------|
| `train/` | 60 mil | 60 mil (uma por imagem) |
| `test/` | 10 mil | 10 mil (uma por imagem) |

Na pasta deste arquivo, com o `uv` já usado nas atividades anteriores:

```text
uv sync
uv run python gerar_big_mnist.py --mnist_txt mnist_txt --out big_mnist_txt
```

`uv sync` uma vez. O `pyproject.toml` é o mesmo ambiente do `train_v2.py` (torch, numpy, tqdm). O gerador usa o `tqdm` para a barra de progresso.

O conjunto cheio em texto ocupa vários GB. Depois de converter para `.emat`, o `.txt` pode ser apagado.

Não use estes `.txt` como entrada direta do laço de treino em Python.

O `train_v2.py` desta pasta é o da Atividade 6. Ele ainda lê imagem 28×28 e pesos 10×784. Adaptá-lo ao 128×128 faz parte do projeto.


Cursor foi usado para formatação e scripts auxiliares
