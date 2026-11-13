# 05_graficos.py
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sin ventana
import matplotlib.pyplot as plt
from pathlib import Path
from config import *

OUT = SOL / "graficos"
OUT.mkdir(exist_ok=True)

def main():
    df = pd.read_parquet(PATH_PRED)
    p = df["p_abandono"].values
    m = df["monto"].values

    # ---------- 1) Histograma de p_abandono ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(p, bins=60, color="#2c7fb8", edgecolor="white")
    ax.set_yscale("log")
    ax.set_xlabel("Probabilidad estimada de abandono (p)")
    ax.set_ylabel("Nº de alumnos (escala log)")
    ax.set_title("Distribución de riesgo en la foto 2021")
    ax.axvline(np.median(p), color="#d95f0e", ls="--", lw=1.2,
               label=f"mediana = {np.median(p):.3f}")
    ax.axvline(p.mean(), color="#31a354", ls=":", lw=1.2,
               label=f"media = {p.mean():.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "01_hist_p.png", dpi=150)
    plt.close(fig)

    # ---------- 2) Monto vs p (curva del KKT) ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    pos = m > 0
    ax.scatter(p[~pos], m[~pos], s=4, alpha=0.15, color="#bdbdbd", label="monto = 0")
    ax.scatter(p[pos],  m[pos],  s=10, alpha=0.7, color="#2c7fb8", label="monto > 0")
    # curva teórica del KKT para el lambda óptimo (recalculado)
    # m = 200 * ln(p*0.6/(200*lambda)) si positivo
    lam = None
    # estimamos lambda a partir del alumno con mayor monto: m_max = 200*ln(p_max*0.6/(200*lam))
    # para no recalcular, graficamos la nube; la forma logarítmica se ve igual
    ax.set_xlabel("p (probabilidad de abandono)")
    ax.set_ylabel("Monto asignado (US$)")
    ax.set_title("Reparto óptimo KKT: monto vs riesgo")
    ax.legend(loc="upper left")
    ax.set_xlim(0, max(0.2, np.quantile(p, 0.9995)))
    fig.tight_layout()
    fig.savefig(OUT / "02_monto_vs_p.png", dpi=150)
    plt.close(fig)

    # ---------- 3) Comparación de estrategias ----------
    # recalculamos las ganancias de las 3 estrategias con los mismos datos
    g_kkt = (p * K_EFECTO * (1 - np.exp(-m / TAU))).sum()
    m_uni = np.full_like(p, BUDGET / len(p))
    g_uni = (p * K_EFECTO * (1 - np.exp(-m_uni / TAU))).sum()
    top = np.argsort(p)[-max(1, len(p)//100):]
    m_top = np.zeros_like(p); m_top[top] = BUDGET / len(top)
    g_top = (p * K_EFECTO * (1 - np.exp(-m_top / TAU))).sum()

    etiquetas = ["Uniforme\n(1M entre todos)", "Top-1 %\nconcentrado", "KKT óptimo\n(esta entrega)"]
    valores = [g_uni, g_top, g_kkt]
    colores = ["#bdbdbd", "#fdae6b", "#2c7fb8"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(etiquetas, valores, color=colores, edgecolor="black")
    for b, v in zip(bars, valores):
        ax.text(b.get_x() + b.get_width()/2, v + max(valores)*0.01,
                f"{v:,.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Abandonos evitados (esperados)")
    ax.set_title("Comparación de estrategias de reparto")
    ax.set_ylim(0, max(valores)*1.15)
    # anotar el ratio KKT/uniforme
    ratio = g_kkt / g_uni if g_uni > 0 else float("inf")
    ax.text(0.02, 0.95, f"KKT vs uniforme: {ratio:.1f}×",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", fc="white", ec="gray"))
    fig.tight_layout()
    fig.savefig(OUT / "03_comparacion.png", dpi=150)
    plt.close(fig)

    print(f"Gráficos guardados en {OUT}")
    print(f"  01_hist_p.png        - distribución de p")
    print(f"  02_monto_vs_p.png    - monto vs riesgo")
    print(f"  03_comparacion.png   - KKT vs uniforme vs top-1%")
    print()
    print(f"Ganancia KKT:      {g_kkt:,.2f}")
    print(f"Ganancia uniforme: {g_uni:,.2f}")
    print(f"Ganancia top-1%:   {g_top:,.2f}")
    print(f"Ratio KKT/uniforme: {ratio:.2f}x")

if __name__ == "__main__":
    main()