"""
Parte II — Correção: Bellman-Ford, Floyd-Warshall, Tarjan
Subgrafo gerado via BFS (vizinhança conectada) em vez de amostragem aleatória.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import gc
import time
import random
import pickle
import numpy as np
import scipy.stats as stats
import networkx as nx
import warnings
warnings.filterwarnings("ignore")

EDGE_FILE       = "/shared/com-lj.ungraph.txt"
OUT_PREFIX      = "/shared/parte2"
CHECKPOINT_FILE = f"{OUT_PREFIX}_checkpoint.pkl"
N_RUNS          = 30
SAMPLE_SIZE     = 1_000
SEED            = 42
random.seed(SEED)
np.random.seed(SEED)

print("=" * 65)
print("PARTE II — CORREÇÃO: Bellman-Ford, Floyd-Warshall, Tarjan")
print("=" * 65)

# ── Utilitários ───────────────────────────────────────────────────
def calc_ic(times, alpha=0.05):
    n    = len(times)
    mean = np.mean(times)
    std  = np.std(times, ddof=1)
    se   = std / np.sqrt(n)
    if n >= 30:
        z    = stats.norm.ppf(1 - alpha / 2)
        lo, hi, dist = mean - z*se, mean + z*se, "Normal (Z)"
    else:
        t    = stats.t.ppf(1 - alpha / 2, df=n - 1)
        lo, hi, dist = mean - t*se, mean + t*se, "t-Student"
    return {"mean": mean, "std": std, "lo": lo, "hi": hi, "n": n, "dist": dist}

def print_ic(res):
    print(f"  Média  : {res['mean']*1000:.4f} ms")
    print(f"  Desvio : {res['std']*1000:.4f} ms")
    print(f"  IC 95% : [{res['lo']*1000:.4f}, {res['hi']*1000:.4f}] ms  ({res['dist']}, n={res['n']})")

# ── Carrega checkpoint existente ──────────────────────────────────
with open(CHECKPOINT_FILE, "rb") as f:
    ckpt = pickle.load(f)

results      = ckpt["results"]
aplicavel    = ckpt["aplicavel"]
complexidade = ckpt["complexidade"]
escopo       = ckpt["escopo"]
print(f"\n  Checkpoint carregado — algoritmos já prontos: {', '.join(results.keys())}")

# ── Carrega grafo CPU ─────────────────────────────────────────────
print("\n[0] Carregando grafo CPU...")
t0 = time.time()
G_cpu = nx.read_edgelist(
    EDGE_FILE, comments="#", nodetype=int, data=False, create_using=nx.Graph()
)
G_cpu.remove_edges_from(nx.selfloop_edges(G_cpu))
nodes_list = list(G_cpu.nodes())
print(f"  {G_cpu.number_of_nodes():,} nós  [{time.time()-t0:.1f}s]")

# ── Subgrafo via BFS — garante conectividade ──────────────────────
print(f"\n  Montando subgrafo conectado via BFS (alvo: {SAMPLE_SIZE} nós)...")
seed_node  = random.choice(nodes_list)
bfs_result = nx.single_source_shortest_path_length(G_cpu, seed_node, cutoff=4)
bfs_nodes  = list(bfs_result.keys())[:SAMPLE_SIZE]

# Se BFS não trouxe nós suficientes, aumenta cutoff
if len(bfs_nodes) < SAMPLE_SIZE:
    bfs_result = nx.single_source_shortest_path_length(G_cpu, seed_node)
    bfs_nodes  = list(bfs_result.keys())[:SAMPLE_SIZE]

H_cpu  = G_cpu.subgraph(bfs_nodes).copy()
H_nodes = list(H_cpu.nodes())
print(f"  Subgrafo: {H_cpu.number_of_nodes()} nós | {H_cpu.number_of_edges()} arestas")
print(f"  Conectado: {nx.is_connected(H_cpu)}")

# ══════════════════════════════════════════════════════════════════
# 5. BELLMAN-FORD
# ══════════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("5. Bellman-Ford (CORRIGIDO)")
print("─"*65)
print("  Complexidade teórica: O(V · E)")
print(f"  Subgrafo: {H_cpu.number_of_nodes()} nós | {H_cpu.number_of_edges()} arestas")

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
escopo["Bellman-Ford"]       = f"Subgrafo {H_cpu.number_of_nodes()} nós BFS (CPU)"

# ══════════════════════════════════════════════════════════════════
# 6. FLOYD-WARSHALL
# ══════════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("6. Floyd-Warshall (CORRIGIDO)")
print("─"*65)
print("  Complexidade teórica: O(V³)")
print(f"  Subgrafo: {H_cpu.number_of_nodes()} nós | {H_cpu.number_of_edges()} arestas")

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
escopo["Floyd-Warshall"]       = f"Subgrafo {H_cpu.number_of_nodes()} nós BFS (CPU)"

# ══════════════════════════════════════════════════════════════════
# 7. TARJAN
# ══════════════════════════════════════════════════════════════════
print("\n" + "─"*65)
print("7. Tarjan (CORRIGIDO)")
print("─"*65)
print("  Complexidade teórica: O(V + E)")
print(f"  Subgrafo DiGraph: {H_cpu.number_of_nodes()} nós")

DG = nx.DiGraph(H_cpu)
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
escopo["Tarjan"]       = f"Subgrafo {DG.number_of_nodes()} nós BFS (DiGraph CPU)"

# ── Salva checkpoint atualizado ───────────────────────────────────
with open(CHECKPOINT_FILE, "wb") as f:
    pickle.dump({"results": results, "aplicavel": aplicavel,
                 "complexidade": complexidade, "escopo": escopo}, f)
print("\n  Checkpoint atualizado.")

# ── Tabela resumo final ───────────────────────────────────────────
print("\n" + "=" * 90)
print("TABELA RESUMO — Parte II (COMPLETA)")
print("=" * 90)

header = f"{'Algoritmo':<16} {'Aplicável':<10} {'Complexidade':<18} {'Escopo':<35} {'Média(ms)':>10} {'IC 95%(ms)':<32} {'Dist.'}"
print(header)
print("-" * 125)

for alg in results:
    res     = results[alg]
    mean_ms = res["mean"] * 1000
    lo_ms   = res["lo"]   * 1000
    hi_ms   = res["hi"]   * 1000
    ic_str  = f"[{lo_ms:.3f}, {hi_ms:.3f}]"
    print(f"{alg:<16} {aplicavel[alg]:<10} {complexidade[alg]:<18} {escopo[alg]:<35} {mean_ms:>10.3f} {ic_str:<32} {res['dist']}")

print("\nFIM")
