"""
Driver — Parte III — Q3: Robustez (Apêndice C)  [isolamento por processo]

Cada medição (baseline / cada repetição aleatória / remoção de hubs) roda
em um SUBPROCESSO Python separado (worker.py). Isso garante que toda a
memória da GPU seja devolvida ao driver CUDA quando o subprocesso termina,
eliminando os problemas de OOM/fragmentação observados ao reaproveitar o
mesmo processo Python para 30+ repetições.

Tem checkpoint em JSON: se a execução for interrompida, rodar de novo
retoma de onde parou (não refaz o que já está pronto).

Uso:
  python driver.py
"""
import os
import sys
import json
import time
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORKER      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker.py")
OUT_PREFIX  = "/shared/parte3"
STATE_FILE  = "/shared/parte3_q3_state.json"
T_RUNS      = 30
MAX_RETRIES = 3
RETRY_SLEEP = 5


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"baseline": None, "random": {}, "hubs": None}


def save_state(state):
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)


def run_worker(mode, rep=-1):
    cmd = [sys.executable, WORKER, "--mode", mode, "--rep", str(rep)]
    for tentativa in range(1 + MAX_RETRIES):
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.time() - t0
        result_line = None
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT|"):
                result_line = line
        if result_line:
            _, m, r, N, Nr, A, B, C, D = result_line.split("|")
            return {"N": int(N), "N_remove": int(Nr), "A": float(A),
                     "B": int(B), "C": float(C), "D": float(D), "t": dt}
        print(f"  [aviso] worker mode={mode} rep={rep} falhou "
              f"(tentativa {tentativa+1}/{MAX_RETRIES+1}, {dt:.1f}s).")
        if proc.stderr.strip():
            print("    stderr (últimas linhas):")
            for ln in proc.stderr.strip().splitlines()[-6:]:
                print(f"      {ln}")
        time.sleep(RETRY_SLEEP)
    raise RuntimeError(f"Worker mode={mode} rep={rep} falhou após {MAX_RETRIES+1} tentativas.")


def main():
    print("=" * 65)
    print("PARTE III — Q3: ROBUSTEZ (Apêndice C) — isolamento por processo")
    print(f"  Início: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    state = load_state()

    # ── Baseline ──
    if state["baseline"] is None:
        print("\n  [Baseline]")
        state["baseline"] = run_worker("baseline")
        save_state(state)
    b = state["baseline"]
    print(f"  A={b['A']:.4f}  B={b['B']:,}  C={b['C']:.4f}  D={b['D']:.6f}  [{b['t']:.1f}s]")

    # ── (a) Remoção aleatória, T=30 ──
    print(f"\n  [a] Remoção ALEATÓRIA {T_RUNS} repetições...")
    for i in range(T_RUNS):
        key = str(i)
        if key in state["random"]:
            continue
        r = run_worker("random", rep=i)
        state["random"][key] = r
        save_state(state)
        print(f"  Rep {i+1:2d}/{T_RUNS}: A={r['A']:.4f} B={r['B']:,} "
              f"C={r['C']:.3f} D={r['D']:.5f}  [{r['t']:.1f}s]")

    A_rand = [state["random"][str(i)]["A"] for i in range(T_RUNS)]
    B_rand = [state["random"][str(i)]["B"] for i in range(T_RUNS)]
    C_rand = [state["random"][str(i)]["C"] for i in range(T_RUNS)]
    D_rand = [state["random"][str(i)]["D"] for i in range(T_RUNS)]

    print(f"\n  Resultados (média ± dp):")
    print(f"  A: {np.mean(A_rand):.4f} ± {np.std(A_rand):.4f}")
    print(f"  B: {np.mean(B_rand):.1f} ± {np.std(B_rand):.1f}")
    print(f"  C: {np.mean(C_rand):.4f} ± {np.std(C_rand):.4f}")
    print(f"  D: {np.mean(D_rand):.6f} ± {np.std(D_rand):.6f}")

    # ── (b) Remoção direcionada (hubs) ──
    if state["hubs"] is None:
        print(f"\n  [b] Remoção DIRECIONADA (top 5% por grau)...")
        state["hubs"] = run_worker("hubs")
        save_state(state)
    h = state["hubs"]
    print(f"  A={h['A']:.4f} (Δ={h['A']-b['A']:+.4f})  "
          f"B={h['B']:,} (Δ={h['B']-b['B']:+,})  "
          f"C={h['C']:.4f} (Δ={h['C']-b['C']:+.4f})  D={h['D']:.6f}  [{h['t']:.1f}s]")

    print(f"""
  INTERPRETAÇÃO (Apêndice C):
  S_rand={np.mean(A_rand):.3f}  vs  S_cent={h['A']:.3f}
  c_cent={h['B']:,}  vs  c_rand={np.mean(B_rand):.0f}

  S_rand >> S_cent e c_cent >> c_rand:
  → rede MUITO VULNERÁVEL a ataques direcionados.
  → padrão típico de redes com lei de potência (scale-free).
  → robusta a falhas aleatórias, frágil a ataques nos hubs.
""")

    # ── Figura ──
    print("\nGerando visualizações Q3...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("LiveJournal — Parte III — Q3 Robustez (Apêndice C)", fontweight="bold")
    for ax_i, (data, title, baseline_v, hub_v) in enumerate([
        (A_rand, "Métrica A: Tamanho LCC",    b["A"], h["A"]),
        (B_rand, "Métrica B: Nº Componentes", b["B"], h["B"]),
        (C_rand, "Métrica C: Distâncias",     b["C"], h["C"]),
    ]):
        ax = axes[ax_i]
        ax.boxplot([data], positions=[1], widths=0.4,
                   patch_artist=True, boxprops=dict(facecolor="#2980b9", alpha=0.7))
        ax.scatter([2], [hub_v], color="#e74c3c", s=100, zorder=5, label=f"Hubs={hub_v:.3f}")
        ax.axhline(baseline_v, color="green", linestyle="--", lw=1.5, label=f"Baseline={baseline_v:.3f}")
        ax.set_xticks([1, 2]); ax.set_xticklabels(["Aleatório\n(T=30)", "Hubs\n(degree)"])
        ax.set_title(title); ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{OUT_PREFIX}_q3_robustez.png", dpi=150, bbox_inches="tight")
    print(f"  Figura salva: {OUT_PREFIX}_q3_robustez.png")

    # ── Tabela resumo ──
    print("\n" + "=" * 70)
    print("TABELA RESUMO — Q3 Robustez")
    print("=" * 70)
    rows = [
        ("── Baseline ──", "", ""),
        ("A — LCC fração", f"{b['A']:.4f}", "Exato GPU"),
        ("B — Nº componentes", f"{b['B']:,}", "Exato GPU"),
        ("C — AvgPath", f"{b['C']:.4f}", "Amostragem"),
        ("D — Fração isolados", f"{b['D']:.6f}", "Exato GPU"),
        ("── Remoção ALEATÓRIA (T=30) ──", "", ""),
        ("A — LCC (média ± dp)", f"{np.mean(A_rand):.4f} ± {np.std(A_rand):.4f}", "30 reps"),
        ("B — Componentes (média ± dp)", f"{np.mean(B_rand):.1f} ± {np.std(B_rand):.1f}", "30 reps"),
        ("C — AvgPath (média ± dp)", f"{np.mean(C_rand):.4f} ± {np.std(C_rand):.4f}", "30 reps"),
        ("D — Isolados (média ± dp)", f"{np.mean(D_rand):.5f} ± {np.std(D_rand):.5f}", "30 reps"),
        ("── Remoção HUBS (degree) ──", "", ""),
        ("A — LCC", f"{h['A']:.4f}  (Δ={h['A']-b['A']:+.4f})", "Exato GPU"),
        ("B — Componentes", f"{h['B']:,}  (Δ={h['B']-b['B']:+,})", "Exato GPU"),
        ("C — AvgPath", f"{h['C']:.4f}  (Δ={h['C']-b['C']:+.4f})", "Amostragem"),
        ("D — Isolados", f"{h['D']:.6f}", "Exato GPU"),
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


if __name__ == "__main__":
    main()
