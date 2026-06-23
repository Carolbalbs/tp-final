# Análise Estrutural e Algorítmica da Rede Social LiveJournal

Este repositório contém o código-fonte, scripts, notebooks e o relatório final do trabalho prático da disciplina **MATA53 - Teoria dos Grafos (2026.1)** na Universidade Federal da Bahia (UFBA).

> 📄 **Relatório em PDF:** Você pode fazer o download da versão formatada em PDF do artigo final aqui: **[relatorio.pdf](https://carolbalbs.github.io/tp-final/224115861.pdf)**
>
> 🌐 **Jupyter Book / Site:** Acesse a versão interativa em: **<https://carolbalbs.github.io/tp-final/>**

---

## Estrutura do Repositório

- [224115861.md](224115861.md) e [paper.qmd](paper.qmd) — Relatório final em formato Markdown/Quarto.
- `docs/` — Arquivos do site estático gerado para o GitHub Pages.
- [data/](data/) — Conjunto de dados tratados (lista de adjacências de comunidades, checkpoints e estados).
  - `data/images/` — Gráficos e visualizações gerados durante a análise.
- [scripts/](scripts/) — Scripts Python automatizados para processamento de alto desempenho na GPU.
  - `scripts/logs/` — Logs de execução detalhados obtidos no servidor de experimentação.
- [referencias.bib](referencias.bib) — Arquivo BibTeX com as referências bibliográficas do relatório.
- [requirements.txt](requirements.txt) — Dependências de bibliotecas Python.

---

## Ambiente Computacional

As análises pesadas e os benchmarks dos algoritmos foram rodados em um servidor dedicado na infraestrutura do **BAMBU Lab (UFBA)** com as seguintes configurações:

### Hardware
- **Processador Gráfico:** 2x NVIDIA RTX A4000 (16 GB de VRAM cada)
- **Servidor:** `bambu-server3`

### Software (Container Podman)
- **Sistema Operacional:** Linux (Ubuntu 22.04)
- **NVIDIA CUDA:** 12.6
- **Driver NVIDIA:** 560.35.03
- **Python:** 3.10
- **Virtualenv:** `/opt/grafos-env/`

---

# 1.Instruções de Execução

## 2. Criação do Container
 
```bash
# Cria o container com suporte a GPU
podman run -d \
  --name snap_analysis \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  -v /shared:/shared \
  python:3.11-slim \
  sleep infinity
 
# Confirma que o container está rodando
podman ps
```
 
Verificar acesso à GPU dentro do container:
```bash
podman exec -it snap_analysis nvidia-smi
```
 
---
 
## 3. Configuração do Ambiente Virtual
 
As bibliotecas cuDF e cuGraph são incompatíveis com TensorFlow (conflito de versão do NumPy). Por isso, criou-se um ambiente virtual isolado:
 
```bash
podman exec -it snap_analysis bash -c "
python3 -m venv /opt/grafos-env
"
```
 
### 3.1 Instalação das dependências
 
```bash
podman exec -it snap_analysis bash -c "
/opt/grafos-env/bin/pip install --upgrade pip && \
/opt/grafos-env/bin/pip install \
    'numpy>=2.0' \
    cudf-cu12 \
    cugraph-cu12 \
    networkx \
    matplotlib \
    scipy \
    --extra-index-url https://pypi.nvidia.com
"
```
 
### 3.2 Verificação do ambiente
 
```bash
podman exec -it snap_analysis /opt/grafos-env/bin/python -c "
import numpy;     print('numpy    :', numpy.__version__)
import cudf;      print('cudf     : OK')
import cugraph;   print('cugraph  : OK')
import networkx;  print('networkx :', networkx.__version__)
import scipy;     print('scipy    :', scipy.__version__)
import matplotlib; print('matplotlib:', matplotlib.__version__)
"
```
 
### 3.3 Tabela de bibliotecas utilizadas
 
| Biblioteca | Uso | Onde roda |
|---|---|---|
| `cudf` | Leitura de CSV e manipulação de DataFrames na GPU | GPU |
| `cugraph` | BFS, SSSP, componentes conexas, triângulos, MST | GPU |
| `networkx` | DFS, Eulerianidade, Bellman-Ford, Floyd-Warshall, Tarjan, Kruskal | CPU |
| `numpy` | Cálculos estatísticos (média, desvio padrão, IC) | CPU |
| `scipy.stats` | Distribuições Normal e t-Student para IC 95% | CPU |
| `matplotlib` | Geração de gráficos e figuras | CPU |
| `collections` | Contagem de frequências (distribuição de graus) | CPU |
| `pickle` | Checkpoint de resultados entre scripts | CPU |
| `gc` | Coleta de lixo explícita para liberar memória GPU | CPU |
| `os` | Configuração de `CUDA_VISIBLE_DEVICES` | CPU |
| `random` | Amostragem aleatória de nós | CPU |
| `time` | Medição de tempo de execução (`time.perf_counter`) | CPU |
| `warnings` | Supressão de avisos não críticos | CPU |
 
---

### 4. **Iniciar o Jupyter:**
   ```bash
   jupyter notebook
   ```
   Abra os notebooks `parte1_all.ipynb` ou `parte1_ungraph.ipynb` no navegador e execute as células.
