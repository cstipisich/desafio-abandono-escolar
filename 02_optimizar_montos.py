import numpy as np, pandas as pd
from scipy.optimize import brentq
from config import *

def montos_optimos(p, budget=BUDGET, k=K_EFECTO, tau=TAU):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0)

    def montos_para_lambda(lam):
        arg = p * k / (tau * lam)
        return np.where(arg > 1.0, tau * np.log(arg), 0.0)

    def f(lam):
        return montos_para_lambda(lam).sum() - budget

    lam_max = float(np.max(p)) * k / tau * (1 - 1e-9)
    lam_min = 1e-6
    for _ in range(60):
        if f(lam_min) > 0:
            break
        lam_min /= 10
    else:
        raise RuntimeError("No se pudo acotar lambda; revisar el presupuesto.")

    lam = brentq(f, lam_min, lam_max, xtol=1e-14, rtol=1e-14, maxiter=1000)
    m = montos_para_lambda(lam)

    resid = budget - m.sum()
    pos = m > 0
    if pos.any() and abs(resid) > 1e-6:
        m[pos] += resid / pos.sum()
        m = np.clip(m, 0.0, None)
    return m, lam

def main():
    df = pd.read_parquet(PATH_PRED)
    p = df["p_abandono"].values
    m, lam = montos_optimos(p)
    df["monto"] = m
    df.to_parquet(PATH_PRED, index=False)

    n_pos = int((m > 0).sum())
    print(f"lambda óptimo: {lam:.6e}")
    print(f"Alumnos con monto > 0: {n_pos:,} / {len(m):,}")
    print(f"Suma de montos: {m.sum():,.2f}  (objetivo {BUDGET:,.0f})")
    if n_pos:
        print(f"Monto mínimo positivo: {m[m>0].min():,.2f}")
        print(f"Mediana de positivos:  {np.median(m[m>0]):,.2f}")
        print(f"Monto máximo:          {m.max():,.2f}")

    g = p * K_EFECTO * (1 - np.exp(-m / TAU))
    print(f"Ganancia esperada (abandonos evitados): {g.sum():,.2f}")

    m_uni = np.full_like(p, BUDGET / len(p))
    g_uni = p * K_EFECTO * (1 - np.exp(-m_uni / TAU))
    print(f"Ganancia si fuera uniforme:            {g_uni.sum():,.2f}")

    top = np.argsort(p)[-max(1, len(p)//100):]
    m_top = np.zeros_like(p)
    m_top[top] = BUDGET / len(top)
    g_top = p * K_EFECTO * (1 - np.exp(-m_top / TAU))
    print(f"Ganancia si fuera top-1% concentrado:  {g_top.sum():,.2f}")

if __name__ == "__main__":
    main()
