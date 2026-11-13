import pandas as pd
from config import *

def main():
    ent = pd.read_csv(PATH_ENTREGA)
    ids21 = set(pd.read_csv(PATH_2021, usecols=["id_alumno"])["id_alumno"])

    assert list(ent.columns) == ["id_alumno", "monto"], f"Columnas: {list(ent.columns)}"
    assert ent["monto"].ge(0).all(), "Hay montos negativos"
    assert ent["monto"].sum() <= BUDGET + 1e-6, f"Suma {ent['monto'].sum()} > {BUDGET}"
    assert len(ent) == len(ent["id_alumno"].unique()), "IDs duplicados"
    assert set(ent["id_alumno"]).issubset(ids21), "IDs fuera de la foto 2021"

    print("OK validación")
    print(f"Filas: {len(ent):,}")
    print(f"Suma:  {ent['monto'].sum():,.2f}")
    print(f"Con monto > 0: {(ent['monto']>0).sum():,}")
    print(f"Máximo: {ent['monto'].max():,.2f}")

if __name__ == "__main__":
    main()
