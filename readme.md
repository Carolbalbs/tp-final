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

## Instruções de Execução

### 1. Execução Remota no Servidor (Podman)
Os scripts foram disparados em segundo plano dentro de um container Podman (`projeto_odonto`) com acesso mapeado à GPU 1. 

Para rodar os experimentos principais no servidor:
```bash
# Executa o script driver no container em segundo plano redirecionando logs
podman exec -d projeto_odonto bash -c \
  "/opt/grafos-env/bin/python /shared/driver.py > /shared/driver.log 2>&1"

# Acompanha a execução em tempo real pelos logs
podman exec -it projeto_odonto tail -f /shared/driver.log
```

### 2. Execução Local dos Notebooks (.ipynb)
Os arquivos `parte1_all.ipynb` e `parte1_ungraph.ipynb` estão disponíveis na raiz do projeto. 

> ⚠️ **Nota sobre dependências de GPU:** A biblioteca `cugraph` e `cudf` (RAPIDS) exigem um ambiente Linux com placas NVIDIA e drivers CUDA configurados. Se você não possuir uma GPU NVIDIA localmente, poderá rodar os códigos usando os blocos de fallback baseados em CPU com a biblioteca **NetworkX**.

#### Passo a Passo para Instalação Local:

1. **Clonar o Repositório:**
   ```bash
   git clone https://github.com/Carolbalbs/tp-final.git
   cd tp-final
   ```

2. **Criar e Ativar Ambiente Virtual:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Instalar Dependências:**
   - **Apenas CPU (NetworkX e processamento comum):**
     ```bash
     pip install pandas scipy networkx matplotlib jupyter
     ```
   - **Com suporte a GPU (NVIDIA RAPIDS - cuGraph e cuDF):**
     Siga as instruções oficiais do [NVIDIA RAPIDS Installer](https://rapids.ai/start.html) usando `conda` ou `mamba` (recomendado para gerenciar pacotes CUDA). Exemplo via conda:
     ```bash
     conda create -n rapids-24.xx -c rapidsai -c conda-forge -c nvidia \
         cugraph cudf python=3.10 cuda-version=12.6
     conda activate rapids-24.xx
     pip install -r requirements.txt
     ```

4. **Iniciar o Jupyter:**
   ```bash
   jupyter notebook
   ```
   Abra os notebooks `parte1_all.ipynb` ou `parte1_ungraph.ipynb` no navegador e execute as células.
