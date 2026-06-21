"""
Parte III — Q3: Robustez (Apêndice C)
Dataset: LiveJournal Social Network (SNAP) — Grafo Completo

Métricas A, B, C, D:
  (a) Remoção aleatória — T=30 repetições → boxplot
  (b) Remoção dos 5% mais centrais (degree) — ranking fixo
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # GPU 0

import gc
import time
import random
import numpy as np
import cudf
import cugraph
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

EDGE_FILE  = "/shared/com-lj.ungraph.txt"
OUT_PREFIX = "/shared/parte3"
T_RUNS     = 30
SEED       = 42
random.seed(SEED)
np.random.seed(SEED)

print("=" * 65)
print("PARTE III — Q3: ROBUSTEZ (Apêndice C) — GPU 0")
print(f"  Início: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 65)

print("\n[0] Carregando grafo...")
t0 = time.time()
edges = cudf.read_csv(
    EDGE_FILE, comment="#", sep="\t", header=None,
    names=["src", "dst"], dtype={"src": "int32", "dst": "int32"}
)
edges = edges[edges["src"] != edges["dst"]].reset_index(drop=True)
G = cugraph.Graph()
G.from_cudf_edgelist(edges, source="src", destination="dst", renumber=True)
N            = G.number_of_vertices()
nodes_list   = G.nodes().to_pandas().tolist()
deg_df       = G.degree()
REMOVAL_FRAC = 0.05
N_remove     = int(REMOVAL_FRAC * N)
print(f"  {N:,} nós | n_remove={N_remove:,}  [{time.time()-t0:.1f}s]")

# ══════════════════════════════════════════════════════════════════
# Q3 — ROBUSTEZ (Apêndice C)
# Métricas A, B, C, D após remoção
# (a) Aleatória: T=30 repetições → boxplot
# (b) Direcionada (degree): ranking fixo, 1 execução
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("Q3. ROBUSTEZ (Apêndice C)")
print("═"*65)

REMOVAL_FRAC = 0.05
N_remove     = int(REMOVAL_FRAC * N)
print(f"  n={N:,}  r={N_remove:,} (5%)  T={T_RUNS} repetições (aleatório)")

def metricas_robustez(edges_df, total_n, k_avg_path=100):
    """
    Retorna métricas A, B, C, D após remoção.
    A: tamanho LCC / n
    B: número de componentes
    C: comprimento médio (amostragem BFS)
    D: fração nós isolados
    """
    if len(edges_df) == 0:
        return 0.0, total_n, float("inf"), 1.0

    Gr  = cugraph.Graph()
    Gr.from_cudf_edgelist(edges_df.reset_index(drop=True),
                           source="src", destination="dst", renumber=True)
    cc      = cugraph.connected_components(Gr)
    sizes   = cc["labels"].value_counts().to_pandas()  # converte para pandas
    lcc_sz  = int(sizes.max())
    n_comp  = int(len(sizes))
    n_isol  = int((sizes == 1).sum())
    frac_A  = lcc_sz / total_n
    frac_D  = n_isol / total_n

    # Comprimento médio via amostragem BFS na LCC
    lcc_label = int(sizes.idxmax())
    cc_pandas = cc.to_pandas()
    lcc_nodes = cc_pandas[cc_pandas["labels"] == lcc_label]["vertex"].tolist()
    samp      = random.sample(lcc_nodes, min(k_avg_path, len(lcc_nodes)))
    paths     = []
    for v in samp:
        bfs   = cugraph.bfs(Gr, start=v)
        # Filtra distâncias > 0 (exclui self) e converte para pandas
        d_ser = bfs["distance"].to_pandas()
        d_val = d_ser[(d_ser > 0) & (d_ser < 999999)].values
        if len(d_val) > 0:
            paths.append(float(d_val.mean()))
        del bfs, d_ser, d_val
        gc.collect()

    avg_path = float(np.mean(paths)) if paths else float("inf")

    del Gr, cc, cc_pandas, sizes
    gc.collect()
    return frac_A, n_comp, avg_path, frac_D

# ── Baseline ──────────────────────────────────────────────────────
print("\n  [Baseline]")
t0 = time.time()
A0, B0, C0, D0 = metricas_robustez(edges, N)
print(f"  A (LCC fração)  : {A0:.4f}")
print(f"  B (componentes) : {B0:,}")
print(f"  C (avg path)    : {C0:.4f}")
print(f"  D (isolados)    : {D0:.6f}  [{time.time()-t0:.1f}s]")

# ── (a) Remoção ALEATÓRIA — T=30 repetições ───────────────────────
print(f"\n  [a] Remoção ALEATÓRIA {T_RUNS} repetições...")
A_rand, B_rand, C_rand, D_rand = [], [], [], []

for i in range(T_RUNS):
    t0       = time.time()
    rem      = cudf.Series(random.sample(nodes_list, N_remove), dtype="int32")
    mask     = ~(edges["src"].isin(rem) | edges["dst"].isin(rem))
    edges_r  = edges[mask]
    A, B, C, D = metricas_robustez(edges_r, N)
    A_rand.append(A); B_rand.append(B)
    C_rand.append(C); D_rand.append(D)
    del rem, mask, edges_r; gc.collect()
    print(f"  Rep {i+1:2d}/{T_RUNS}: A={A:.4f} B={B:,} C={C:.3f} D={D:.5f}  [{time.time()-t0:.1f}s]")

print(f"\n  Resultados (média ± dp):")
print(f"  A: {np.mean(A_rand):.4f} ± {np.std(A_rand):.4f}")
print(f"  B: {np.mean(B_rand):.1f} ± {np.std(B_rand):.1f}")
print(f"  C: {np.mean(C_rand):.4f} ± {np.std(C_rand):.4f}")
print(f"  D: {np.mean(D_rand):.6f} ± {np.std(D_rand):.6f}")

# ── (b) Remoção DIRECIONADA (degree centrality) ───────────────────
print(f"\n  [b] Remoção DIRECIONADA (top {N_remove:,} por grau)...")
t0    = time.time()
hubs  = deg_df.sort_values("degree", ascending=False).head(N_remove)["vertex"]
mask  = ~(edges["src"].isin(hubs) | edges["dst"].isin(hubs))
edges_h = edges[mask]
A_h, B_h, C_h, D_h = metricas_robustez(edges_h, N)
del hubs, mask, edges_h; gc.collect()

print(f"  A (LCC fração)  : {A_h:.4f}  (Δ={A_h-A0:+.4f})")
print(f"  B (componentes) : {B_h:,}  (Δ={B_h-B0:+,})")
print(f"  C (avg path)    : {C_h:.4f}  (Δ={C_h-C0:+.4f})")
print(f"  D (isolados)    : {D_h:.6f}  [{time.time()-t0:.1f}s]")

print(f"""
  INTERPRETAÇÃO (Apêndice C):
  S_rand={np.mean(A_rand):.3f}  vs  S_cent={A_h:.3f}
  c_cent={B_h:,}  vs  c_rand={np.mean(B_rand):.0f}

  S_rand >> S_cent e c_cent >> c_rand:
  → rede MUITO VULNERÁVEL a ataques direcionados.
  → padrão típico de redes com lei de potência (scale-free).
  → robusta a falhas aleatórias, frágil a ataques nos hubs.
""")

# ══════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════
# VISUALIZAÇÕES — Q3 Robustez
# ══════════════════════════════════════════════════════════════════
print("\nGerando visualizações Q3...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("LiveJournal — Parte III — Q3 Robustez (Apêndice C)", fontweight="bold")

for ax_i, (data, label, title, baseline) in enumerate([
    (A_rand, "A — LCC fração",    "Métrica A: Tamanho LCC",      A0),
    (B_rand, "B — Componentes",   "Métrica B: Nº Componentes",   B0),
    (C_rand, "C — Avg Path",      "Métrica C: Distâncias",       C0),
]):
    ax = axes[ax_i]
    ax.boxplot([data], positions=[1], widths=0.4,
               patch_artist=True, boxprops=dict(facecolor="#2980b9", alpha=0.7))
    hub_val = [A_h, B_h, C_h][ax_i]
    ax.scatter([2], [hub_val], color="#e74c3c", s=100, zorder=5, label=f"Hubs={hub_val:.3f}")
    ax.axhline(baseline, color="green", linestyle="--", lw=1.5, label=f"Baseline={baseline:.3f}")
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Aleatório\n(T=30)", "Hubs\n(degree)"])
    ax.set_title(title); ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(f"{OUT_PREFIX}_q3_robustez.png", dpi=150, bbox_inches="tight")
print(f"  Figura salva: {OUT_PREFIX}_q3_robustez.png")

print("\n" + "=" * 70)
print("TABELA RESUMO — Q3 Robustez")
print("=" * 70)
rows = [
    ("── Baseline ──",                   "",                                           ""),
    ("A — LCC fração",                   f"{A0:.4f}",                                 "Exato GPU"),
    ("B — Nº componentes",               f"{B0:,}",                                   "Exato GPU"),
    ("C — AvgPath",                      f"{C0:.4f}",                                 "Amostragem"),
    ("D — Fração isolados",              f"{D0:.6f}",                                 "Exato GPU"),
    ("── Remoção ALEATÓRIA (T=30) ──",   "",                                           ""),
    ("A — LCC (média ± dp)",             f"{np.mean(A_rand):.4f} ± {np.std(A_rand):.4f}", "30 reps"),
    ("B — Componentes (média ± dp)",     f"{np.mean(B_rand):.1f} ± {np.std(B_rand):.1f}", "30 reps"),
    ("C — AvgPath (média ± dp)",         f"{np.mean(C_rand):.4f} ± {np.std(C_rand):.4f}", "30 reps"),
    ("D — Isolados (média ± dp)",        f"{np.mean(D_rand):.5f} ± {np.std(D_rand):.5f}", "30 reps"),
    ("── Remoção HUBS (degree) ──",      "",                                           ""),
    ("A — LCC",                          f"{A_h:.4f}  (Δ={A_h-A0:+.4f})",            "Exato GPU"),
    ("B — Componentes",                  f"{B_h:,}  (Δ={B_h-B0:+,})",                "Exato GPU"),
    ("C — AvgPath",                      f"{C_h:.4f}  (Δ={C_h-C0:+.4f})",            "Amostragem"),
    ("D — Isolados",                     f"{D_h:.6f}",                                "Exato GPU"),
]
print(f"{'Métrica':<38} {'Valor':<35} {'Tipo'}")
print("-" * 78)
for m, v, t in rows:
    if v == "":
        print(f"\n{m}")
    else:
        print(f"  {m:<36} {v:<35} {t}")

print(f"\n  Fim: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Figura: {OUT_PREFIX}_q3_robustez.png")
