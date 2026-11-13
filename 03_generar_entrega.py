import pandas as pd, numpy as np
from config import *

def main():
    df = pd.read_parquet(PATH_PRED)[["id_alumno", "monto"]].copy()
    df["monto"] = df["monto"].clip(lower=0).round(2)

    resid = round(BUDGET - df["monto"].sum(), 2)
    if abs(resid) > 0:
        idx = df["monto"].idxmax()
        df.loc[idx, "monto"] = round(df.loc[idx, "monto"] + resid, 2)

    df = df.sort_values("id_alumno").reset_index(drop=True)
    df.to_csv(PATH_ENTREGA, index=False)
    print(f"Entrega escrita en {PATH_ENTREGA}")
    print(f"Filas: {len(df):,}   Suma: {df['monto'].sum():,.2f}")
    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
