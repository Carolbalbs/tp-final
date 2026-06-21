"""
Parte II — Algoritmos da Disciplina
Dataset: LiveJournal Social Network (SNAP) — Grafo Completo

Algoritmos:
  1. BFS — Busca em Largura
  2. DFS — Busca em Profundidade
  3. Verificação de Eulerianidade
  4. Dijkstra
  5. Bellman-Ford
  6. Floyd-Warshall
  7. Algoritmo de Tarjan
  8. Prim — Árvore Geradora Mínima
  9. Kruskal — Árvore Geradora Mínima

Para cada algoritmo:
  - Complexidade teórica
  - Aplicabilidade no dataset
  - Tempo real: média, desvio padrão, IC 95% (Normal se n>=30, t-Student se n<30)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import gc
import time
import random
import pickle
import numpy as np
import scipy.stats as stats
import cudf
import cugraph
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

EDGE_FILE   = "/shared/com-lj.ungraph.txt"
OUT_PREFIX  = "/shared/parte2"
N_RUNS      = 30       # repetições para IC (Normal se >=30)
SAMPLE_SIZE = 1_000    # nós para subgrafo em algoritmos pesados
SEED        = 42
random.seed(SEED)
np.random.seed(SEED)

print("=" * 65)
print("PARTE II — ALGORITMOS DA DISCIPLINA (GPU)")
print("=" * 65)
print(f"  Repetições (N_RUNS)  : {N_RUNS}")
print(f"  Subgrafo pesados     : {SAMPLE_SIZE} nós")

# ══════════════════════════════════════════════════════════════════
# UTILITÁRIOS
# ══════════════════════════════════════════════════════════════════

def calc_ic(times, alpha=0.05):
    n    = len(times)
    mean = np.mean(times)
    std  = np.std(times, ddof=1)
    se   = std / np.sqrt(n)
    if n >= 30:
        z    = stats.norm.ppf(1 - alpha / 2)
        lo   = mean - z * se
        hi   = mean + z * se
        dist = "Normal (Z)"
    else:
        t    = stats.t.ppf(1 - alpha / 2, df=n - 1)
        lo   = mean - t * se
        hi   = mean + t * se
        dist = "t-Student"
    return {"mean": mean, "std": std, "lo": lo, "hi": hi, "n": n, "dist": dist}

def print_ic(res):
    print(f"  Média  : {res['mean']*1000:.4f} ms")
    print(f"  Desvio : {res['std']*1000:.4f} ms")
    print(f"  IC 95% : [{res['lo']*1000:.4f}, {res['hi']*1000:.4f}] ms  ({res['dist']}, n={res['n']})")

results      = {}
aplicavel    = {}
complexidade = {}
escopo       = {}

CHECKPOINT_FILE = f"{OUT_PREFIX}_checkpoint.pkl"

def save_checkpoint():
    """Salva o estado atual em disco. Chamada após cada algoritmo."""
    try:
        with open(CHECKPOINT_FILE, "wb") as f:
            pickle.dump({
                "results":      results,
                "aplicavel":    aplicavel,
                "complexidade": complexidade,
                "escopo":       escopo,
            }, f)
    except Exception as e:
        print(f"  [checkpoint] AVISO: falha ao salvar ({e})")

def load_checkpoint():
    """Carrega checkpoint anterior, se existir, e popula os dicts globais."""
    if not os.path.exists(CHECKPOINT_FILE):
        return
    try:
        with open(CHECKPOINT_FILE, "rb") as f:
            data = pickle.load(f)
        results.update(data.get("results", {}))
        aplicavel.update(data.get("aplicavel", {}))
        complexidade.update(data.get("complexidade", {}))
        escopo.update(data.get("escopo", {}))
        if results:
            print(f"\n[checkpoint] Retomando — já concluídos: {', '.join(results.keys())}")
    except Exception as e:
        print(f"  [checkpoint] AVISO: falha ao carregar, ignorando ({e})")

load_checkpoint()

# ══════════════════════════════════════════════════════════════════
# CARREGAMENTO
# ══════════════════════════════════════════════════════════════════
print("\n[0] Carregando grafos...")
t0 = time.time()

# Grafo completo na GPU
edges_gpu = cudf.read_csv(
    EDGE_FILE, comment="#", sep="\t", header=None,
    names=["src", "dst"], dtype={"src": "int32", "dst": "int32"}
)
edges_gpu = edges_gpu[edges_gpu["src"] != edges_gpu["dst"]].reset_index(drop=True)

G_gpu = cugraph.Graph()
G_gpu.from_cudf_edgelist(edges_gpu, source="src", destination="dst", renumber=True)

# Grafo completo GPU ponderado (peso=1.0 em todas as arestas — grafo original
# não tem pesos). Reaproveitado por Dijkstra (sssp exige grafo ponderado) e
# por Prim (minimum_spanning_tree).
edges_w = edges_gpu.copy()
edges_w["weight"] = 1.0
G_w = cugraph.Graph()
G_w.from_cudf_edgelist(edges_w, source="src", destination="dst",
                        edge_attr="weight", renumber=True)
del edges_w
gc.collect()

# Grafo completo na CPU (para DFS, Euler, BF, FW, Tarjan, Prim, Kruskal)
G_cpu = nx.read_edgelist(
    EDGE_FILE, comments="#", nodetype=int, data=False, create_using=nx.Graph()
)
G_cpu.remove_edges_from(nx.selfloop_edges(G_cpu))

nodes_list = list(G_cpu.nodes())

# Subgrafo CPU para algoritmos pesados
sample_nodes = random.sample(nodes_list, SAMPLE_SIZE)
H_cpu = G_cpu.subgraph(sample_nodes).copy()
H_cpu = H_cpu.subgraph(max(nx.connected_components(H_cpu), key=len)).copy()
H_nodes = list(H_cpu.nodes())

print(f"  Grafo completo GPU : {G_gpu.number_of_vertices():,} nós | {G_gpu.number_of_edges():,} arestas")
print(f"  Grafo completo CPU : {G_cpu.number_of_nodes():,} nós")
print(f"  Subgrafo CPU       : {H_cpu.number_of_nodes()} nós | {H_cpu.number_of_edges()} arestas")
print(f"  Carregamento       : {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════════
# 1. BFS — GPU
# ══════════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("1. BFS — Breadth-First Search")
print("─"*65)
print("  Complexidade teórica: O(V + E)")
print("  Aplicável           : SIM — grafo completo GPU")

if "BFS" in results:
    print("  [checkpoint] já concluído — pulando.")
else:
    try:
        times = []
        src_nodes = random.sample(nodes_list, N_RUNS)
        for v in src_nodes:
            t0 = time.perf_counter()
            _ = cugraph.bfs(G_gpu, start=v)
            times.append(time.perf_counter() - t0)
            gc.collect()

        res = calc_ic(times)
        print_ic(res)
        results["BFS"]      = res
        aplicavel["BFS"]    = "SIM"
        complexidade["BFS"] = "O(V + E)"
        escopo["BFS"]       = "Grafo completo (GPU)"
    except Exception as e:
        print(f"  ERRO: {e}")
        aplicavel["BFS"]    = "ERRO"
        complexidade["BFS"] = "O(V + E)"
        escopo["BFS"]       = "Grafo completo (GPU)"
    save_checkpoint()

# ══════════════════════════════════════════════════════════════════
# 2. DFS — CPU (cugraph não tem DFS nativo)
# ══════════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("2. DFS — Depth-First Search")
print("─"*65)
print("  Complexidade teórica: O(V + E)")
print("  Aplicável           : SIM — grafo completo CPU")
print("  Nota: cuGraph não implementa DFS — usando NetworkX")

if "DFS" in results:
    print("  [checkpoint] já concluído — pulando.")
else:
    try:
        times = []
        for v in random.sample(nodes_list, N_RUNS):
            t0 = time.perf_counter()
            _ = list(nx.dfs_edges(G_cpu, source=v))
            times.append(time.perf_counter() - t0)

        res = calc_ic(times)
        print_ic(res)
        results["DFS"]      = res
        aplicavel["DFS"]    = "SIM"
        complexidade["DFS"] = "O(V + E)"
        escopo["DFS"]       = "Grafo completo (CPU)"
    except Exception as e:
        print(f"  ERRO: {e}")
        aplicavel["DFS"]    = "ERRO"
        complexidade["DFS"] = "O(V + E)"
        escopo["DFS"]       = "Grafo completo (CPU)"
    save_checkpoint()

# ══════════════════════════════════════════════════════════════════
# 3. EULERIANIDADE — CPU
# ══════════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("3. Verificação de Eulerianidade")
print("─"*65)
print("  Complexidade teórica: O(V + E)")
print("  Aplicável           : SIM")

if "Eulerianidade" in results:
    print("  [checkpoint] já concluído — pulando.")
else:
    try:
        times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            has_circuit = nx.is_eulerian(G_cpu)
            has_path    = nx.has_eulerian_path(G_cpu)
            times.append(time.perf_counter() - t0)

        odd_degree = sum(1 for _, d in G_cpu.degree() if d % 2 != 0)
        print(f"  Circuito Euleriano : {has_circuit}")
        print(f"  Caminho Euleriano  : {has_path}")
        print(f"  Vértices grau ímpar: {odd_degree:,}")
        print(f"  → Não é Euleriano (esperado em redes sociais)")

        res = calc_ic(times)
        print_ic(res)
        results["Eulerianidade"]      = res
        aplicavel["Eulerianidade"]    = "SIM"
        complexidade["Eulerianidade"] = "O(V + E)"
        escopo["Eulerianidade"]       = "Grafo completo (CPU)"
    except Exception as e:
        print(f"  ERRO: {e}")
        aplicavel["Eulerianidade"]    = "ERRO"
        complexidade["Eulerianidade"] = "O(V + E)"
        escopo["Eulerianidade"]       = "Grafo completo (CPU)"
    save_checkpoint()

# ══════════════════════════════════════════════════════════════════
# 4. DIJKSTRA — GPU
# ══════════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("4. Dijkstra")
print("─"*65)
print("  Complexidade teórica: O((V + E) log V)")
print("  Aplicável           : SIM — sem pesos negativos")
print("  Nota: grafo não ponderado → todos os pesos = 1.0 (cugraph.sssp exige grafo ponderado)")

if "Dijkstra" in results:
    print("  [checkpoint] já concluído — pulando.")
else:
    try:
        times = []
        for v in random.sample(nodes_list, N_RUNS):
            t0 = time.perf_counter()
            _ = cugraph.sssp(G_w, source=v)
            times.append(time.perf_counter() - t0)
            gc.collect()

        res = calc_ic(times)
        print_ic(res)
        results["Dijkstra"]      = res
        aplicavel["Dijkstra"]    = "SIM"
        complexidade["Dijkstra"] = "O((V+E) log V)"
        escopo["Dijkstra"]       = "Grafo completo (GPU)"
    except Exception as e:
        print(f"  ERRO: {e}")
        aplicavel["Dijkstra"]    = "ERRO"
        complexidade["Dijkstra"] = "O((V+E) log V)"
        escopo["Dijkstra"]       = "Grafo completo (GPU)"
    save_checkpoint()

# ══════════════════════════════════════════════════════════════════
# 5. BELLMAN-FORD — subgrafo CPU
# ══════════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("5. Bellman-Ford")
print("─"*65)
print("  Complexidade teórica: O(V · E)")
print(f"  Aplicável           : LIMITADO — O(V·E) inviável no grafo completo")
print(f"  → Executado em subgrafo de {H_cpu.number_of_nodes()} nós")

if "Bellman-Ford" in results:
    print("  [checkpoint] já concluído — pulando.")
else:
    try:
        times = []
        for v in random.sample(H_nodes, min(N_RUNS, len(H_nodes))):
            t0 = time.perf_counter()
            _ = nx.single_source_bellman_ford_path_length(H_cpu, v)
            times.append(time.perf_counter() - t0)

        res = calc_ic(times)
        print_ic(res)
        results["Bellman-Ford"]      = res
        aplicavel["Bellman-Ford"]    = "LIMITADO"
        complexidade["Bellman-Ford"] = "O(V · E)"
        escopo["Bellman-Ford"]       = f"Subgrafo {H_cpu.number_of_nodes()} nós (CPU)"
    except Exception as e:
        print(f"  ERRO: {e}")
        aplicavel["Bellman-Ford"]    = "ERRO"
        complexidade["Bellman-Ford"] = "O(V · E)"
        escopo["Bellman-Ford"]       = f"Subgrafo {H_cpu.number_of_nodes()} nós (CPU)"
    save_checkpoint()

# ══════════════════════════════════════════════════════════════════
# 6. FLOYD-WARSHALL — subgrafo CPU
# ══════════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("6. Floyd-Warshall")
print("─"*65)
print("  Complexidade teórica: O(V³)")
print(f"  Aplicável           : LIMITADO — O(V³) inviável no grafo completo")
print(f"  → Executado em subgrafo de {H_cpu.number_of_nodes()} nós")

if "Floyd-Warshall" in results:
    print("  [checkpoint] já concluído — pulando.")
else:
    try:
        times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            _ = dict(nx.floyd_warshall(H_cpu))
            times.append(time.perf_counter() - t0)

        res = calc_ic(times)
        print_ic(res)
        results["Floyd-Warshall"]      = res
        aplicavel["Floyd-Warshall"]    = "LIMITADO"
        complexidade["Floyd-Warshall"] = "O(V³)"
        escopo["Floyd-Warshall"]       = f"Subgrafo {H_cpu.number_of_nodes()} nós (CPU)"
    except Exception as e:
        print(f"  ERRO: {e}")
        aplicavel["Floyd-Warshall"]    = "ERRO"
        complexidade["Floyd-Warshall"] = "O(V³)"
        escopo["Floyd-Warshall"]       = f"Subgrafo {H_cpu.number_of_nodes()} nós (CPU)"
    save_checkpoint()

# ══════════════════════════════════════════════════════════════════
# 7. TARJAN — subgrafo DiGraph CPU
# ══════════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("7. Algoritmo de Tarjan (SCCs)")
print("─"*65)
print("  Complexidade teórica: O(V + E)")
print("  Aplicável           : ADAPTADO — LiveJournal é não-dirigido")
print("  → Executado em versão dirigida do subgrafo")

DG = nx.DiGraph(H_cpu)

if "Tarjan" in results:
    print("  [checkpoint] já concluído — pulando.")
else:
    try:
        times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            sccs = list(nx.strongly_connected_components(DG))
            times.append(time.perf_counter() - t0)

        print(f"  SCCs encontradas: {len(sccs)}")
        print(f"  Maior SCC       : {max(len(s) for s in sccs)} nós")

        res = calc_ic(times)
        print_ic(res)
        results["Tarjan"]      = res
        aplicavel["Tarjan"]    = "ADAPTADO"
        complexidade["Tarjan"] = "O(V + E)"
        escopo["Tarjan"]       = f"Subgrafo {DG.number_of_nodes()} nós (DiGraph CPU)"
    except Exception as e:
        print(f"  ERRO: {e}")
        aplicavel["Tarjan"]    = "ERRO"
        complexidade["Tarjan"] = "O(V + E)"
        escopo["Tarjan"]       = f"Subgrafo {DG.number_of_nodes()} nós (DiGraph CPU)"
    save_checkpoint()

# ══════════════════════════════════════════════════════════════════
# 8. PRIM — GPU (spanning tree)
# ══════════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("8. Prim — Árvore Geradora Mínima")
print("─"*65)
print("  Complexidade teórica: O(E log V)")
print("  Aplicável           : SIM — grafo completo GPU")
print("  Nota: sem pesos → todos pesos = 1 (qualquer spanning tree) — reaproveita G_w")

if "Prim" in results:
    print("  [checkpoint] já concluído — pulando.")
else:
    try:
        times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            _ = cugraph.minimum_spanning_tree(G_w)
            times.append(time.perf_counter() - t0)
            gc.collect()

        res = calc_ic(times)
        print_ic(res)
        results["Prim"]      = res
        aplicavel["Prim"]    = "SIM"
        complexidade["Prim"] = "O(E log V)"
        escopo["Prim"]       = "Grafo completo (GPU)"
    except Exception as e:
        print(f"  ERRO: {e}")
        aplicavel["Prim"]    = "ERRO"
        complexidade["Prim"] = "O(E log V)"
        escopo["Prim"]       = "Grafo completo (GPU)"
    save_checkpoint()

# ══════════════════════════════════════════════════════════════════
# 9. KRUSKAL — CPU (cugraph MST usa Borůvka internamente)
# ══════════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("9. Kruskal — Árvore Geradora Mínima")
print("─"*65)
print("  Complexidade teórica: O(E log E)")
print("  Aplicável           : SIM — grafo completo CPU")

if "Kruskal" in results:
    print("  [checkpoint] já concluído — pulando.")
else:
    try:
        times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            _ = nx.minimum_spanning_tree(G_cpu, algorithm="kruskal")
            times.append(time.perf_counter() - t0)

        res = calc_ic(times)
        print_ic(res)
        results["Kruskal"]      = res
        aplicavel["Kruskal"]    = "SIM"
        complexidade["Kruskal"] = "O(E log E)"
        escopo["Kruskal"]       = "Grafo completo (CPU)"
    except Exception as e:
        print(f"  ERRO: {e}")
        aplicavel["Kruskal"]    = "ERRO"
        complexidade["Kruskal"] = "O(E log E)"
        escopo["Kruskal"]       = "Grafo completo (CPU)"
    save_checkpoint()

# ══════════════════════════════════════════════════════════════════
# VISUALIZAÇÃO
# ══════════════════════════════════════════════════════════════════
print("\nGerando visualização...")

fig, ax = plt.subplots(figsize=(13, 5))
algs      = list(results.keys())
means_ms  = [results[a]["mean"] * 1000 for a in algs]
err_lo    = [results[a]["mean"] * 1000 - results[a]["lo"] * 1000 for a in algs]
err_hi    = [results[a]["hi"]  * 1000 - results[a]["mean"] * 1000 for a in algs]
colors    = []
for a in algs:
    if escopo[a].endswith("(GPU)"):
        colors.append("#2980b9")
    elif aplicavel[a] == "LIMITADO":
        colors.append("#e74c3c")
    else:
        colors.append("#27ae60")

ax.bar(algs, means_ms, color=colors, alpha=0.85, edgecolor="white")
ax.errorbar(algs, means_ms, yerr=[err_lo, err_hi],
            fmt="none", color="black", capsize=5, linewidth=1.5)
ax.set_ylabel("Tempo médio (ms)")
ax.set_yscale("log")
ax.set_title("Parte II — Tempo médio dos algoritmos com IC 95%", fontweight="bold")
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.tick_params(axis="x", rotation=20)

from matplotlib.patches import Patch
legend = [
    Patch(color="#2980b9", label="Grafo completo (GPU)"),
    Patch(color="#27ae60", label="Grafo completo (CPU)"),
    Patch(color="#e74c3c", label=f"Subgrafo {SAMPLE_SIZE} nós (CPU)"),
]
ax.legend(handles=legend)

plt.tight_layout()
plt.savefig(f"{OUT_PREFIX}_algoritmos.png", dpi=150, bbox_inches="tight")
print(f"  Figura salva: {OUT_PREFIX}_algoritmos.png")

# ══════════════════════════════════════════════════════════════════
# TABELA RESUMO
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("TABELA RESUMO — Parte II")
print("=" * 90)

header = f"{'Algoritmo':<16} {'Aplicável':<10} {'Complexidade':<18} {'Escopo':<30} {'Média(ms)':>10} {'IC 95%(ms)':<30} {'Dist.'}"
print(header)
print("-" * 120)

for alg in results:
    res    = results[alg]
    mean_ms = res["mean"] * 1000
    lo_ms   = res["lo"]   * 1000
    hi_ms   = res["hi"]   * 1000
    ic_str  = f"[{lo_ms:.3f}, {hi_ms:.3f}]"
    print(f"{alg:<16} {aplicavel[alg]:<10} {complexidade[alg]:<18} {escopo[alg]:<30} {mean_ms:>10.3f} {ic_str:<30} {res['dist']}")

print(f"\nFIM — figura em {OUT_PREFIX}_algoritmos.png")
