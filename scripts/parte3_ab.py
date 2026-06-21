"""
Parte III — Análise Estrutural
Dataset: LiveJournal Social Network (SNAP) — Grafo Completo

Seguindo Apêndices A, B e C do enunciado.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import gc
import time
import random
import numpy as np
import scipy.stats as stats
import cudf
import cugraph
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import collections
import warnings
warnings.filterwarnings("ignore")

EDGE_FILE  = "/shared/com-lj.ungraph.txt"
OUT_PREFIX = "/shared/parte3"
K_SAMPLE   = 500   # amostragem BFS
T_RUNS     = 30    # repetições remoção aleatória (Apêndice C)
SEED       = 42
random.seed(SEED)
np.random.seed(SEED)

print("=" * 65)
print("PARTE III — ANÁLISE ESTRUTURAL (GPU)")
print(f"  Início: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 65)

# ──────────────────────────────────────────────────────────────────
# CARREGAMENTO
# ──────────────────────────────────────────────────────────────────
print("\n[0] Carregando grafo...")
t0 = time.time()

edges = cudf.read_csv(
    EDGE_FILE, comment="#", sep="\t", header=None,
    names=["src", "dst"], dtype={"src": "int32", "dst": "int32"}
)
edges = edges[edges["src"] != edges["dst"]].reset_index(drop=True)

G = cugraph.Graph()
G.from_cudf_edgelist(edges, source="src", destination="dst", renumber=True)

N          = G.number_of_vertices()
E          = G.number_of_edges()
nodes_list = G.nodes().to_pandas().tolist()
deg_df     = G.degree()
deg_pandas = deg_df["degree"].to_pandas()
k_avg      = float(deg_pandas.mean())

print(f"  {N:,} nós | {E:,} arestas  [{time.time()-t0:.1f}s]")

# ══════════════════════════════════════════════════════════════════
# Q1 — SMALL-WORLD (Apêndice A)
# Implementação MANUAL — sem nx.average_clustering nem
# nx.average_shortest_path_length
# Critério: L(G) ≈ L_random  E  C(G) >> C_random
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("Q1. SMALL-WORLD (Apêndice A — implementação manual)")
print("═"*65)

sample = random.sample(nodes_list, K_SAMPLE)

# ── C(v) manual: C(v) = 2*T(v) / (k*(k-1))
# Onde T(v) = triângulos incidentes em v (via cugraph.triangle_count)
# e k = grau de v. Implementação manual da fórmula sem usar
# nx.average_clustering ou qualquer função de clustering pronta.
print(f"\n  [1a] Clustering manual (triangle_count) em {K_SAMPLE} nós...")
t0 = time.time()

# Calcula triângulos por nó e graus — operações primitivas na GPU
tri_df  = cugraph.triangle_count(G)   # {vertex, counts} — triângulos incidentes
deg_ldf = G.degree()                   # {vertex, degree}

tri_df  = tri_df.set_index("vertex")
deg_ldf = deg_ldf.set_index("vertex")
tri_df["degree"] = deg_ldf["degree"]

# Para cada nó amostrado, calcula C(v) = 2*T(v) / (k*(k-1)) manualmente
sample_gpu = cudf.Series(sample, dtype="int32")
tri_sample = tri_df[tri_df.index.isin(sample_gpu)]

# Aplica fórmula elemento a elemento (sem biblioteca de clustering)
mask_k2    = tri_sample["degree"] >= 2
tri_valid  = tri_sample[mask_k2]
clust_vals = (2.0 * tri_valid["counts"]) / (
              tri_valid["degree"] * (tri_valid["degree"] - 1))

# Nós com grau < 2 têm C(v) = 0
C_values = clust_vals.to_pandas().tolist()
C_values += [0.0] * int((~mask_k2).sum())

C_real = float(np.mean(C_values))
del tri_df, deg_ldf, tri_sample, tri_valid, clust_vals, sample_gpu, mask_k2
gc.collect()
print(f"  C_real : {C_real:.6f}  [{time.time()-t0:.1f}s]")

# ── L(G) manual: BFS completo, média das distâncias
print(f"\n  [1b] Comprimento médio manual em {K_SAMPLE} nós...")
t0 = time.time()

L_values = []
for v in sample:
    bfs   = cugraph.bfs(G, start=v)
    dists = bfs["distance"][bfs["distance"] > 0].to_pandas()
    if len(dists) > 0:
        L_values.append(float(dists.mean()))
    del bfs; gc.collect()

L_real = float(np.mean(L_values))
print(f"  L_real : {L_real:.4f}  [{time.time()-t0:.1f}s]")

# ── Grafo aleatório equivalente (Erdős–Rényi)
C_random = k_avg / N
L_random = np.log(N) / np.log(k_avg)
ratio_C  = C_real / C_random
ratio_L  = L_real / L_random
sigma    = ratio_C / ratio_L

print(f"\n  C_random (ER)   : {C_random:.8f}")
print(f"  L_random (ER)   : {L_random:.4f}")
print(f"  C_real/C_random : {ratio_C:.2f}x  (deve ser >> 1)")
print(f"  L_real/L_random : {ratio_L:.4f}   (deve ser ≈ 1)")
print(f"  sigma           : {sigma:.2f}  ({'SMALL-WORLD ✓' if sigma > 1 else 'NÃO small-world'})")

# ══════════════════════════════════════════════════════════════════
# Q2 — LEI DE POTÊNCIA (Apêndice B)
# Verifica P(k) ~ k^(-gamma)
# Plota histograma + CCDF em escala log-log
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("Q2. LEI DE POTÊNCIA (Apêndice B)")
print("═"*65)

deg_arr   = deg_pandas.values.astype(float)
deg_count = collections.Counter(deg_pandas.tolist())
ks_all    = np.array(sorted(deg_count.keys()), dtype=float)
pks_all   = np.array([deg_count[k] for k in ks_all.astype(int)], dtype=float)
pks_all   = pks_all / pks_all.sum()

# CCDF: P(K >= k)
ccdf_vals = np.array([np.sum(pks_all[ks_all >= k]) for k in ks_all])

# Regressão OLS log-log
K_MIN  = 10
mask   = ks_all >= K_MIN
log_k  = np.log10(ks_all[mask])
log_pk = np.log10(pks_all[mask])
slope, intercept, r_value, p_value, _ = stats.linregress(log_k, log_pk)
gamma_ols = -slope

# MLE Hill
tail      = deg_arr[deg_arr >= K_MIN]
gamma_mle = 1 + len(tail) / np.sum(np.log(tail / (K_MIN - 0.5)))
is_power  = (2.0 <= gamma_ols <= 4.0) and (r_value**2 > 0.85)

print(f"  Gamma OLS (log-log, k>={K_MIN}): {gamma_ols:.3f}")
print(f"  R²                            : {r_value**2:.4f}")
print(f"  p-value                       : {p_value:.2e}")
print(f"  Gamma MLE (Hill)              : {gamma_mle:.3f}")
print(f"  CONCLUSÃO: {'Lei de potência ✓ — scale-free' if is_power else 'Não confirmada'}")


# ══════════════════════════════════════════════════════════════════
# VISUALIZAÇÕES — Q1 e Q2
# ══════════════════════════════════════════════════════════════════
print("\nGerando visualizações Q1 e Q2...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("LiveJournal — Parte III — Q1 e Q2", fontweight="bold")

ax = axes[0]
ax.scatter(ks_all, pks_all, s=4, alpha=0.5, color="#2980b9", label="P(k)")
k_fit  = np.logspace(np.log10(K_MIN), np.log10(ks_all.max()), 200)
pk_fit = 10**intercept * k_fit**slope
ax.plot(k_fit, pk_fit, "r-", lw=2, label=f"γ={gamma_ols:.2f}, R²={r_value**2:.3f}")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("k"); ax.set_ylabel("P(k)")
ax.set_title("Lei de Potência — PDF log-log")
ax.legend(); ax.grid(True, which="both", linestyle="--", alpha=0.3)

ax2 = axes[1]
ax2.scatter(ks_all, ccdf_vals, s=4, alpha=0.5, color="#8e44ad")
ax2.set_xscale("log"); ax2.set_yscale("log")
ax2.set_xlabel("k"); ax2.set_ylabel("P(K ≥ k)")
ax2.set_title("CCDF log-log")
ax2.grid(True, which="both", linestyle="--", alpha=0.3)

ax3 = axes[2]
ax3.hist(C_values, bins=30, color="#27ae60", alpha=0.8, edgecolor="white")
ax3.axvline(C_real,   color="red",  linestyle="--", lw=2, label=f"C_real={C_real:.4f}")
ax3.axvline(C_random, color="blue", linestyle="--", lw=2, label=f"C_random={C_random:.2e}")
ax3.set_xlabel("C(v)"); ax3.set_ylabel("Frequência")
ax3.set_title(f"Clustering manual (k={K_SAMPLE})")
ax3.legend()

plt.tight_layout()
plt.savefig(f"{OUT_PREFIX}_q1q2.png", dpi=150, bbox_inches="tight")
print(f"  Figura salva: {OUT_PREFIX}_q1q2.png")

print("\n" + "=" * 65)
print("TABELA RESUMO — Q1 e Q2")
print("=" * 65)
rows = [
    ("── Q1. SMALL-WORLD ──",      "",                           ""),
    ("C_real (manual k=500)",      f"{C_real:.6f}",              "Amostragem"),
    ("C_random (Erdős–Rényi)",     f"{C_random:.8f}",            "Teórico"),
    ("C_real / C_random",          f"{ratio_C:.2f}x",            ""),
    ("L_real (manual k=500)",      f"{L_real:.4f}",              "Amostragem"),
    ("L_random (Erdős–Rényi)",     f"{L_random:.4f}",            "Teórico"),
    ("L_real / L_random",          f"{ratio_L:.4f}",             ""),
    ("Sigma",                      f"{sigma:.2f}",               "sigma>1 → SW"),
    ("── Q2. LEI DE POTÊNCIA ──",  "",                           ""),
    ("Gamma OLS",                  f"{gamma_ols:.3f}",           "Exato"),
    ("Gamma MLE (Hill)",           f"{gamma_mle:.3f}",           "Exato"),
    ("R²",                         f"{r_value**2:.4f}",          ""),
    ("Scale-free",                 "SIM" if is_power else "NÃO", ""),
]
print(f"{'Métrica':<38} {'Valor':<25} {'Tipo'}")
print("-" * 70)
for m, v, t in rows:
    if v == "":
        print(f"\n{m}")
    else:
        print(f"  {m:<36} {v:<25} {t}")

print(f"\n  Fim: {time.strftime('%Y-%m-%d %H:%M:%S')}")
