"""
Análise Estrutural - Parte I
Dataset: LiveJournal Social Network (SNAP) — Grafo Completo
com-lj.ungraph.txt

Métricas exatas via GPU (cuGraph):
  - Vértices, arestas, graus, densidade
  - Componentes conexas
  - Triângulos e clusterização

Métricas por amostragem BFS (k=500 nós, justificado no relatório):
  - Diâmetro, Raio, Comprimento médio dos caminhos
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import gc
import random
import cudf
import cugraph
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import collections
import time
import warnings
warnings.filterwarnings("ignore")

EDGE_FILE  = "/shared/com-lj.ungraph.txt"
OUT_PREFIX = "/shared/parte1"
K_SAMPLE   = 500    # nós amostrados para diâmetro/raio/avg_path
SEED       = 42
random.seed(SEED)

print("=" * 65)
print("PARTE I — Grafo Completo LiveJournal (GPU 1)")
print("=" * 65)
print(f"  Diâmetro/Raio/AvgPath: amostragem k={K_SAMPLE} nós")
print(f"  Demais métricas      : exatas")

# ──────────────────────────────────────────────────────────────────
# 1. CARREGAMENTO
# ──────────────────────────────────────────────────────────────────
print("\n[1/6] Carregando grafo na GPU...")
t0 = time.time()

edges = cudf.read_csv(
    EDGE_FILE, comment="#", sep="\t", header=None,
    names=["src", "dst"], dtype={"src": "int32", "dst": "int32"}
)
loops_removed = int((edges["src"] == edges["dst"]).sum())
edges = edges[edges["src"] != edges["dst"]].reset_index(drop=True)
print(f"  Auto-loops removidos: {loops_removed}")

G = cugraph.Graph()
G.from_cudf_edgelist(edges, source="src", destination="dst", renumber=True)

n_nodes = G.number_of_vertices()
n_edges = G.number_of_edges()
print(f"  {n_nodes:,} nós | {n_edges:,} arestas  [{time.time()-t0:.1f}s]")

# ──────────────────────────────────────────────────────────────────
# 2. MÉTRICAS BÁSICAS — EXATAS
# ──────────────────────────────────────────────────────────────────
print("\n[2/6] Métricas básicas (exatas)...")
t0 = time.time()

deg_df     = G.degree()
deg_ser    = deg_df["degree"]
deg_min    = int(deg_ser.min())
deg_max    = int(deg_ser.max())
deg_avg    = float(deg_ser.mean())
deg_med    = float(deg_ser.median())
density    = (2 * n_edges) / (n_nodes * (n_nodes - 1))
deg_pandas = deg_ser.to_pandas()
del deg_df, deg_ser
gc.collect()

print(f"  Vértices : {n_nodes:,}")
print(f"  Arestas  : {n_edges:,}")
print(f"  Densidade: {density:.8f}")
print(f"  Grau mín : {deg_min}")
print(f"  Grau máx : {deg_max}")
print(f"  Grau méd : {deg_avg:.4f}")
print(f"  Grau med : {deg_med:.1f}  [{time.time()-t0:.2f}s]")

# ──────────────────────────────────────────────────────────────────
# 3. COMPONENTES CONEXAS — EXATAS
# ──────────────────────────────────────────────────────────────────
print("\n[3/6] Componentes conexas (exatas)...")
t0 = time.time()

cc_df      = cugraph.connected_components(G)
n_comp     = int(cc_df["labels"].nunique())
sizes_ser  = cc_df["labels"].value_counts().sort_values(ascending=False)
comp_sizes = sizes_ser.to_pandas().tolist()

print(f"  Nº componentes : {n_comp:,}")
print(f"  Tamanhos top-5 : {comp_sizes[:5]}  [{time.time()-t0:.2f}s]")

# Monta LCC para BFS
lcc_label     = int(sizes_ser.index[0])
lcc_size      = comp_sizes[0]
lcc_nodes_gpu = cc_df[cc_df["labels"] == lcc_label]["vertex"]

edges_lcc = edges[
    edges["src"].isin(lcc_nodes_gpu) & edges["dst"].isin(lcc_nodes_gpu)
].reset_index(drop=True)

del cc_df, sizes_ser, lcc_nodes_gpu, edges
gc.collect()

G_lcc = cugraph.Graph()
G_lcc.from_cudf_edgelist(edges_lcc, source="src", destination="dst", renumber=True)
del edges_lcc
del G
gc.collect()

print(f"  LCC: {lcc_size:,} nós")

# ──────────────────────────────────────────────────────────────────
# 4. DIÂMETRO, RAIO, COMPRIMENTO MÉDIO — AMOSTRAGEM BFS
# ──────────────────────────────────────────────────────────────────
print(f"\n[4/6] Diâmetro, Raio e Comprimento Médio (amostragem k={K_SAMPLE})...")
print(f"  Justificativa: BFS exato exigiria {lcc_size:,} × O(V+E) ≈ 17 dias de CPU.")
print(f"  Amostragem com k={K_SAMPLE} nós aleatórios — IC 95% reportado na tabela.")

all_nodes  = G_lcc.nodes().to_pandas().tolist()
sample     = random.sample(all_nodes, min(K_SAMPLE, len(all_nodes)))

ecc_list   = []
path_list  = []
t0         = time.time()

for i, v in enumerate(sample):
    bfs       = cugraph.bfs(G_lcc, start=v)
    reachable = bfs["distance"][bfs["distance"] >= 0]
    dists     = reachable.to_pandas()
    ecc_list.append(int(dists.max()))
    path_list.append(float(dists[dists > 0].mean()))
    del bfs, reachable, dists
    gc.collect()

    if i % 50 == 0 and i > 0:
        elapsed = time.time() - t0
        eta     = (elapsed / i) * (len(sample) - i)
        print(f"  {i:>4}/{len(sample)} | {elapsed/60:.1f} min | ETA: {eta/60:.0f} min",
              end="\r", flush=True)

t_bfs = time.time() - t0

# Estatísticas
diam_est = max(ecc_list)
raio_est = min(ecc_list)
avg_path = float(np.mean(path_list))
std_path = float(np.std(path_list, ddof=1))

# IC 95% para comprimento médio (n=500 >= 30 → distribuição Normal)
import scipy.stats as stats
z95  = stats.norm.ppf(0.975)
se   = std_path / np.sqrt(len(path_list))
ic_lo = avg_path - z95 * se
ic_hi = avg_path + z95 * se

print(f"\n  Diâmetro estimado: {diam_est}  (SNAP: 17)")
print(f"  Raio estimado    : {raio_est}")
print(f"  Comprimento médio: {avg_path:.4f}  ± {std_path:.4f}")
print(f"  IC 95% (Normal)  : [{ic_lo:.4f}, {ic_hi:.4f}]")
print(f"  Tempo BFS        : {t_bfs/60:.1f} min")

# ──────────────────────────────────────────────────────────────────
# 5. TRIÂNGULOS e CLUSTERIZAÇÃO — EXATOS
# ──────────────────────────────────────────────────────────────────
print("\n[5/6] Triângulos e Clusterização (exatos)...")

# Recria grafo completo
edges2 = cudf.read_csv(
    EDGE_FILE, comment="#", sep="\t", header=None,
    names=["src", "dst"], dtype={"src": "int32", "dst": "int32"}
)
edges2 = edges2[edges2["src"] != edges2["dst"]].reset_index(drop=True)
G2 = cugraph.Graph()
G2.from_cudf_edgelist(edges2, source="src", destination="dst", renumber=True)
del edges2
gc.collect()

t0      = time.time()
tri_df  = cugraph.triangle_count(G2)
n_tri   = int(tri_df["counts"].sum()) // 3
t_tri   = time.time() - t0
print(f"  Triângulos: {n_tri:,}  (SNAP: 177,820,130)  [{t_tri:.1f}s]")

deg_df3          = G2.degree().set_index("vertex")
tri_df           = tri_df.set_index("vertex")
tri_df["degree"] = deg_df3["degree"]
tri_df           = tri_df[tri_df["degree"] >= 2]
tri_df["clust"]  = (2.0 * tri_df["counts"]) / (
                    tri_df["degree"] * (tri_df["degree"] - 1))
avg_clust        = float(tri_df["clust"].mean())
print(f"  Clustering médio: {avg_clust:.6f}  (SNAP: 0.2843)")

# ──────────────────────────────────────────────────────────────────
# 6. VISUALIZAÇÃO
# ──────────────────────────────────────────────────────────────────
print("\n[6/6] Gerando visualizações...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("LiveJournal — Grafo Completo — Análise Estrutural (Parte I)",
             fontweight="bold")

# Distribuição de graus log-log
ax = axes[0]
deg_count = collections.Counter(deg_pandas.tolist())
xs = sorted(deg_count.keys())
ys = [deg_count[x] for x in xs]
ax.scatter(xs, ys, s=4, alpha=0.5, color="#2980b9")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Grau (log)"); ax.set_ylabel("Frequência (log)")
ax.set_title("Distribuição de Graus (log-log)")
ax.grid(True, which="both", linestyle="--", alpha=0.4)

# Histograma de excentricidades amostradas
ax2 = axes[1]
ax2.hist(ecc_list, bins=range(min(ecc_list), max(ecc_list)+2),
         color="#e67e22", alpha=0.8, edgecolor="white")
ax2.axvline(diam_est, color="red",    linestyle="--", lw=2, label=f"Diâm. est.={diam_est}")
ax2.axvline(raio_est, color="blue",   linestyle="--", lw=2, label=f"Raio est.={raio_est}")
ax2.set_xlabel("Excentricidade"); ax2.set_ylabel("Nº de nós amostrados")
ax2.set_title(f"Excentricidades amostradas (k={K_SAMPLE})")
ax2.legend()

plt.tight_layout()
plt.savefig(f"{OUT_PREFIX}_grafo_completo.png", dpi=150, bbox_inches="tight")
print(f"  Figura salva: {OUT_PREFIX}_grafo_completo.png")

# ── TABELA RESUMO ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TABELA RESUMO — Grafo Completo LiveJournal")
print("=" * 70)

rows = [
    ("Número de vértices",              f"{n_nodes:,}",                    "Exato"),
    ("Número de arestas",               f"{n_edges:,}",                    "Exato"),
    ("Auto-loops removidos",            str(loops_removed),                "Exato"),
    ("Grau mínimo",                     str(deg_min),                      "Exato"),
    ("Grau máximo",                     str(deg_max),                      "Exato"),
    ("Grau médio",                      f"{deg_avg:.4f}",                  "Exato"),
    ("Grau mediano",                    f"{deg_med:.1f}",                  "Exato"),
    ("Densidade",                       f"{density:.8f}",                  "Exato"),
    ("Nº componentes conexas",          f"{n_comp:,}",                     "Exato"),
    ("Tamanho LCC",                     f"{lcc_size:,}",                   "Exato"),
    ("Diâmetro (LCC)",                  f"~{diam_est}  (SNAP: 17)",        f"Amostragem k={K_SAMPLE}"),
    ("Raio (LCC)",                      f"~{raio_est}",                    f"Amostragem k={K_SAMPLE}"),
    ("Comprimento médio (LCC)",         f"{avg_path:.4f} ± {std_path:.4f}",f"Amostragem k={K_SAMPLE}"),
    ("IC 95% comp. médio",              f"[{ic_lo:.4f}, {ic_hi:.4f}]",    "Normal (n≥30)"),
    ("Coef. clusterização médio",       f"{avg_clust:.6f}",                "Exato (GPU)"),
    ("Número de triângulos",            f"{n_tri:,}",                      "Exato (GPU)"),
    ("Tempo BFS (amostragem)",          f"{t_bfs/60:.1f} min",             "—"),
    ("Tempo triângulos",                f"{t_tri:.1f}s",                   "—"),
]

print(f"{'Métrica':<38} {'Valor':<30} {'Tipo'}")
print("-" * 78)
for m, v, t in rows:
    print(f"{m:<38} {v:<30} {t}")

print(f"\nFIM — figura em {OUT_PREFIX}_grafo_completo.png")
