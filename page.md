# Trabalho de Análise de Grafos {.unnumbered}

Este trabalho consiste na análise estrutural e algorítmica de um grafo atribuído, com o objetivo de extrair propriedades fundamentais, executar algoritmos clássicos e investigar fenômenos como small‑world, lei de potência e robustez. O relatório final deve documentar todas as medidas computadas, justificar eventuais ausências e incluir uma tabela resumo.

::: {.callout-note appearance="simple"}
**Organização do relatório:**  
O trabalho está dividido em duas partes obrigatórias (Análise Estrutural e Algoritmos) e uma parte complementar com análises avançadas. Os apêndices fornecem referências e orientações para os tópicos de small‑world, lei de potência e robustez.
:::

---

## Parte I – Análise Estrutural Obrigatória

Para o grafo atribuído, devem ser respondidas as seguintes questões, com todas as medidas apresentadas no relatório (ou justificativa caso não sejam computadas):

- Número de vértices  
- Número de arestas  
- Grau mínimo, máximo e médio  
- Distribuição de graus  
- Densidade  
- Número de componentes conexas  
- Tamanho de cada componente  
- Diâmetro  
- Raio  
- Comprimento médio dos caminhos  
- Coeficiente de clusterização médio  
- Número de triângulos (se viável)  

Além disso, deve ser apresentada uma visualização do grafo (pode ser uma versão reduzida).

**Tabela resumo:**  
Incluir uma tabela indicando quais informações foram computadas e quais não, com as respectivas justificativas.

---

## Parte II – Algoritmos da Disciplina

Devem ser implementados e analisados os seguintes algoritmos:

- Busca em Largura (BFS)  
- Busca em Profundidade (DFS)  
- Verificação de Eulerianidade  
- Dijkstra (se aplicável)  
- Bellman‑Ford (se aplicável)  
- Floyd‑Warshall  
- Algoritmo de Tarjan  
- Algoritmo de Árvore Geradora Mínima (Prim ou Kruskal)  

**Análise de complexidade:**  
Para cada algoritmo, deve‑se apresentar:

- A complexidade teórica (notação assintótica)  
- A complexidade observada (tempo real de execução)  

Para a análise de tempo real, calcular:

- Média  
- Desvio padrão  
- Intervalo de Confiança (IC), conforme a regra:

| Situação | Distribuição usada para calcular IC |
|----------|--------------------------------------|
| \(n \geq 30\) | Normal (Z): \(\bar{x} \pm z \cdot \frac{\sigma}{\sqrt{n}}\) |
| \(n < 30\) e \(\sigma\) desconhecido | t‑Student: \(\bar{x} \pm t \cdot \frac{\sigma}{\sqrt{n}}\) |

Sendo \(z\) o valor crítico da normal e \(t\) o valor crítico da distribuição t.

---

## Análises Complementares

### 1. Propriedade Small‑World

Verificar se o grafo apresenta indícios de small‑world, considerando:

- Comprimento médio dos caminhos \(L(G)\) próximo ao de um grafo aleatório equivalente.  
- Coeficiente de clusterização \(C(G)\) muito maior que o do grafo aleatório.  

**Implementação:**  
Evitar o uso de bibliotecas prontas para o cálculo do small‑world; codificar a comparação com o modelo aleatório.

**Implicação prática:**  
Discutir o significado de uma rede small‑world para o contexto do grafo analisado.

### 2. Lei de Potência (Power Law)

Investigar se a distribuição de graus segue \(P(k) \sim k^{-\gamma}\), com \(\gamma > 1\).

**Procedimento:**

- Calcular a distribuição de graus (frequência de cada valor \(k\)).  
- Construir histograma e/ou a distribuição acumulada complementar (CCDF).  
- Plotar em escala log‑log: \(\log(P(k)) \times \log(k)\) deve exibir comportamento linear.

### 3. Robustez do Grafo

Avaliar a robustez sob duas condições:

**a) Remoção aleatória de 5% dos vértices**  
- Realizar \(T\) repetições (ex.: 30 ou 50).  
- Em cada repetição, escolher \(r = \lceil 0.05 \times |V| \rceil\) vértices uniformemente ao acaso e removê‑los (com arestas incidentes).  
- Calcular as métricas:  
  - A – Tamanho da maior componente  
  - B – Número de componentes  
  - C – Distâncias (ex.: diâmetro ou caminho médio)  
  - D – Fração de nós isolados  
- Reportar boxplots com média ± desvio padrão para cada métrica.

**b) Remoção dos 5% mais centrais (ataque direcionado)**  
- Escolher uma medida de centralidade (recomendado: *betweenness* ou *degree*).  
- Ordenar os vértices pela centralidade (decrescente) e remover os \(r\) primeiros.  
- Calcular as mesmas métricas A, B, C e D.  

**Interpretação:**  
Comparar os resultados dos dois cenários. Se \(S_{\text{rand}} \gg S_{\text{cent}}\) e \(c_{\text{cent}} \gg c_{\text{rand}}\), a rede é vulnerável a ataques direcionados (típico de redes com hubs). Se ambos os cenários degradam de forma semelhante, a rede é mais homogênea.

### 4. Descoberta mais interessante

Cada grupo deve destacar uma descoberta relevante sobre o grafo, relacionada aos resultados obtidos.

### 5. (Extra) Aproximação a modelos clássicos

Avaliar se o grafo se aproxima de algum modelo clássico (Erdős‑Rényi, Barabási‑Albert, Watts‑Strogatz) e justificar com base nas propriedades observadas.

---

## Apêndices de Referência

- **Apêndice A – Small‑World:** [Wikipedia](https://en.wikipedia.org/wiki/Small-world_network) e definição formal com comparação a grafo aleatório.  
- **Apêndice B – Lei de Potência:** [Wikipedia](https://en.wikipedia.org/wiki/Power_law) e procedimento de verificação em escala log‑log.  
- **Apêndice C – Robustez:** Orientações para remoção aleatória e direcionada, escolha de centralidade e métricas a reportar.

---

::: {.callout-warning}
**Observações importantes:**  
- Caso o grafo seja desconexo, definir uma política de análise (ex.: trabalhar na maior componente conexa ou analisar todas as componentes separadamente) e documentá‑la.  
- Para a análise de small‑world, **implementar o código manualmente**, sem recorrer a bibliotecas prontas.  
- Todas as medidas e análises devem ser apresentadas de forma clara no relatório, com tabelas, gráficos e interpretações.
:::