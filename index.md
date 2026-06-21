# Introdução {.unnumbered}

Este trabalho consiste na análise estrutural e algorítmica de um grafo atribuído, com o objetivo de extrair propriedades fundamentais, executar algoritmos clássicos e investigar fenômenos como small‑world, lei de potência e robustez. O relatório final deve documentar todas as medidas computadas, justificar eventuais ausências e incluir uma tabela resumo. O dataset utilizado nesse experimento foi disponibilizado pelo [SNAP - Stanford Network Analysis Project](https://snap.stanford.edu/index.html).

::: {.callout-note appearance="simple"}
> 📄 **Relatório em PDF:** Faça o download do relatório final formatado em PDF aqui: **[relatorio.pdf](224115861.pdf)**.
>
> 💻 **Código-Fonte:** Acesse o repositório completo com scripts e dados em: **<https://github.com/Carolbalbs/tp-final>**
:::

## O Projeto

O grafo atribuído (ID 25) para a matrícula **224115861** corresponde ao conjunto de dados **LiveJournal social network and ground-truth communities**, caracterizado por ser um grafo não direcionado contendo 3.997.962 nós e 34.681.189 arestas. 

Devido à escala multimilionária do grafo, as análises foram executadas no ambiente do **BAMBU Lab (UFBA)** utilizando GPUs **NVIDIA RTX A4000** com a suíte **NVIDIA RAPIDS** (`cuGraph` / `cuDF`), alcançando speedups de até **553×** em relação a execuções sequenciais em CPU.

## Executando Localmente

Se você deseja rodar os notebooks (`parte1_all.ipynb` e `parte1_ungraph.ipynb`) ou scripts localmente:

1. **Clonar o repositório:**
   ```bash
   git clone https://github.com/Carolbalbs/tp-final.git
   cd tp-final
   ```
2. **Instalar dependências de CPU:**
   ```bash
   pip install -r requirements.txt
   ```
   *Nota: O código rodará em modo de compatibilidade com NetworkX se nenhuma GPU NVIDIA estiver disponível.*
3. **Instalar dependências de GPU (Opcional):**
   Para habilitar aceleração por hardware, instale a suíte [NVIDIA RAPIDS](https://rapids.ai) em seu ambiente Linux configurado com CUDA.
4. **Executar o Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

## Créditos

Desenvolvido na Universidade Federal da Bahia por **Ana Carolina Balbino**. Agradecemos ao **BAMBU Lab (UFBA)** pelo suporte computacional e ao **SNAP** pela disponibilização do dataset. Agradecemos também ao professor **Dr. Bruno P. S.** e aos colegas da disciplina **MATA53 - Teoria dos Grafos (2026.1)** pelo incentivo e discussões.
