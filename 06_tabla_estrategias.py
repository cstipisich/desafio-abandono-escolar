# 06_tabla_estrategias.py
"""
Compara estrategias de reparto del millón:
  - uniforme entre todos
  - top-k % concentrado (varios k)
  - KKT óptimo (el de la entrega)
Genera:
  - estrategias.csv
  - estrategias.md  (tabla lista para pegar en el README)
"""
import numpy as np, pandas as pd
from config import *

def ganancia(p, m, k=K_EFECTO, tau=TAU):
    return float((p * k * (1 - np.exp(-m / tau))).sum())

def main():
    df = pd.read_parquet(PATH_PRED)
    p = df["p_abandono"].values.astype(float)
    m_kkt = df["monto"].values.astype(float)
    n = len(p)
    g_kkt = ganancia(p, m_kkt)

    filas = []

    # Uniforme
    m_uni = np.full(n, BUDGET / n)
    filas.append({"estrategia": "Uniforme entre todos", "alumnos_con_plata": n,
                  "monto_por_alumno_mediana": round(float(np.median(m_uni)), 2),
                  "ganancia": round(ganancia(p, m_uni), 2),
                  "ratio_vs_kkt": round(ganancia(p, m_uni) / g_kkt, 3)})

    # Top-k % concentrado
    for pct in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        k = max(1, int(round(n * pct / 100.0)))
        idx = np.argsort(p)[-k:]
        m = np.zeros(n); m[idx] = BUDGET / k
        g = ganancia(p, m)
        filas.append({"estrategia": f"Top-{pct:g} % concentrado",
                      "alumnos_con_plata": k,
                      "monto_por_alumno_mediana": round(float(np.median(m[m > 0])), 2),
                      "ganancia": round(g, 2),
                      "ratio_vs_kkt": round(g / g_kkt, 3)})

    # KKT (la entrega)
    filas.append({"estrategia": "KKT óptimo (entrega)",
                  "alumnos_con_plata": int((m_kkt > 0).sum()),
                  "monto_por_alumno_mediana": round(float(np.median(m_kkt[m_kkt > 0])), 2),
                  "ganancia": round(g_kkt, 2),
                  "ratio_vs_kkt": 1.0})

    tab = pd.DataFrame(filas)
    tab = tab.sort_values("ganancia", ascending=False).reset_index(drop=True)
    tab.to_csv(SOL / "estrategias.csv", index=False)

    # Markdown
    md = ["| Estrategia | Alumnos con plata | Monto mediano (US$) | Abandonos evitados | Ratio vs KKT |",
          "|---|---:|---:|---:|---:|"]
    for _, r in tab.iterrows():
        md.append(f"| {r['estrategia']} | {int(r['alumnos_con_plata']):,} | "
                  f"{r['monto_por_alumno_mediana']:,.2f} | {r['ganancia']:,.2f} | "
                  f"{r['ratio_vs_kkt']:.3f}× |")
    (SOL / "estrategias.md").write_text("\n".join(md), encoding="utf-8")

    print(tab.to_string(index=False))
    print()
    print(f"Guardado: {SOL / 'estrategias.csv'}")
    print(f"Guardado: {SOL / 'estrategias.md'}")

if __name__ == "__main__":
    main()