"""
Worker isolado — Parte III — Q3: Robustez (Apêndice C)

Cada chamada deste script carrega o grafo do ZERO, calcula as métricas
A, B, C, D para UM único cenário (baseline / uma repetição aleatória /
remoção de hubs) e termina. Isso garante que toda a memória da GPU seja
devolvida ao driver CUDA na saída do processo, eliminando os problemas
de fragmentação/teto de pool observados ao reaproveitar o mesmo
processo Python para 30+ repetições.

Uso:
  python worker.py --mode baseline
  python worker.py --mode random --rep 0
  python worker.py --mode hubs
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gc
import random
import argparse
import numpy as np
import cudf
import cugraph
import warnings
warnings.filterwarnings("ignore")

EDGE_FILE    = "/shared/com-lj.ungraph.txt"
REMOVAL_FRAC = 0.05
SEED         = 42
K_AVG_PATH   = 100


def metricas_robustez(edges_df, total_n, k_avg_path=K_AVG_PATH):
    if len(edges_df) == 0:
        return 0.0, total_n, float("inf"), 1.0

    Gr = cugraph.Graph()
    Gr.from_cudf_edgelist(edges_df.reset_index(drop=True),
                           source="src", destination="dst", renumber=True)
    cc      = cugraph.connected_components(Gr)
    sizes   = cc["labels"].value_counts().to_pandas()
    lcc_sz  = int(sizes.max())
    n_comp  = int(len(sizes))
    n_isol  = int((sizes == 1).sum())
    frac_A  = lcc_sz / total_n
    frac_D  = n_isol / total_n

    lcc_label = int(sizes.idxmax())
    cc_pandas = cc.to_pandas()
    lcc_nodes = cc_pandas[cc_pandas["labels"] == lcc_label]["vertex"].tolist()
    samp      = random.sample(lcc_nodes, min(k_avg_path, len(lcc_nodes)))
    paths     = []
    for v in samp:
        bfs   = cugraph.bfs(Gr, start=v)
        d_ser = bfs["distance"].to_pandas()
        d_val = d_ser[(d_ser > 0) & (d_ser < 999999)].values
        if len(d_val) > 0:
            paths.append(float(d_val.mean()))
        del bfs, d_ser, d_val

    avg_path = float(np.mean(paths)) if paths else float("inf")

    del Gr, cc, cc_pandas, sizes, lcc_nodes, samp
    gc.collect()
    return frac_A, n_comp, avg_path, frac_D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "random", "hubs"], required=True)
    ap.add_argument("--rep", type=int, default=-1)
    args = ap.parse_args()

    edges = cudf.read_csv(
        EDGE_FILE, comment="#", sep="\t", header=None,
        names=["src", "dst"], dtype={"src": "int32", "dst": "int32"}
    )
    edges = edges[edges["src"] != edges["dst"]].reset_index(drop=True)
    G = cugraph.Graph()
    G.from_cudf_edgelist(edges, source="src", destination="dst", renumber=True)
    N        = G.number_of_vertices()
    N_remove = int(REMOVAL_FRAC * N)

    if args.mode == "baseline":
        A, B, C, D = metricas_robustez(edges, N)

    elif args.mode == "random":
        # Seed única e determinística por repetição -> reprodutível em reruns
        random.seed(SEED + args.rep)
        nodes_list = G.nodes().to_pandas().tolist()
        rem        = cudf.Series(random.sample(nodes_list, N_remove), dtype="int32")
        mask       = ~(edges["src"].isin(rem) | edges["dst"].isin(rem))
        edges_r    = edges[mask]
        A, B, C, D = metricas_robustez(edges_r, N)

    elif args.mode == "hubs":
        deg_df = G.degree()
        hubs   = deg_df.sort_values("degree", ascending=False).head(N_remove)["vertex"]
        mask   = ~(edges["src"].isin(hubs) | edges["dst"].isin(hubs))
        edges_h = edges[mask]
        A, B, C, D = metricas_robustez(edges_h, N)

    # Única linha que o driver.py vai parsear do stdout
    print(f"RESULT|{args.mode}|{args.rep}|{N}|{N_remove}|{A:.6f}|{B}|{C:.6f}|{D:.6f}")


if __name__ == "__main__":
    main()
