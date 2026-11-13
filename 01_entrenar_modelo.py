import warnings, joblib, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
from config import *

warnings.filterwarnings("ignore")

def cargar_historico(path, anios, etiqueta):
    print(f"Leyendo {path} ...")
    df = pd.read_csv(path, low_memory=False)   # .csv plano, sin gzip
    print(f"  filas totales: {len(df):,}")
    df = df[df["agno"].isin(anios)].copy()
    print(f"  filas en años {anios}: {len(df):,}")
    if etiqueta not in df.columns:
        raise SystemExit(f"ERROR: no existe la columna etiqueta '{etiqueta}'. "
                         f"Columnas disponibles: {list(df.columns)[:50]}...")
    df = df[df[etiqueta].notna()].copy()
    return df

def preparar_X(df):
    cols = [c for c in CAT_COLS + NUM_COLS if c in df.columns]
    faltantes = [c for c in CAT_COLS + NUM_COLS if c not in df.columns]
    if faltantes:
        print(f"  columnas faltantes (se ignoran): {faltantes}")
    X = df[cols].copy()
    for c in CAT_COLS:
        if c in X.columns:
            X[c] = X[c].astype("category")
    for c in NUM_COLS:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X

def main():
    df = cargar_historico(PATH_HIST, ANIOS_TRAIN, COLS_ETIQUETA)
    y = df[COLS_ETIQUETA].astype(int).values
    X = preparar_X(df)
    print(f"Tasa de abandono en train: {y.mean():.4f}  (n={len(y):,})")

    anio_train, anio_val = min(ANIOS_TRAIN), max(ANIOS_TRAIN)
    mask_tr = (df["agno"] == anio_train).values
    mask_va = (df["agno"] == anio_val).values
    X_tr, y_tr = X[mask_tr], y[mask_tr]
    X_va, y_va = X[mask_va], y[mask_va]
    print(f"train: {len(X_tr):,} (año {anio_train})  val: {len(X_va):,} (año {anio_val})")

    base = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=3000,
        learning_rate=0.05,        # Subir un poco
        num_leaves=63,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        # is_unbalance=True,       # <-- COMENTAR O ELIMINAR ESTA LÍNEA
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    base.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
    )
    auc = roc_auc_score(y_va, base.predict_proba(X_va)[:, 1])
    print(f"AUC val ({anio_val}): {auc:.4f}")

    from sklearn.frozen import FrozenEstimator
    calib = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
    calib.fit(X_va, y_va)
    p_va = calib.predict_proba(X_va)[:, 1]
    print(f"Brier calibrado: {brier_score_loss(y_va, p_va):.5f}")

    joblib.dump({"modelo": calib, "features": list(X.columns)}, PATH_MODELO)
    print(f"Modelo guardado en {PATH_MODELO}")

    print(f"Leyendo {PATH_2021} ...")
    d21 = pd.read_csv(PATH_2021, low_memory=False)
    print(f"  filas 2021: {len(d21):,}")
    X21 = preparar_X(d21)
    for c in X.columns:
        if c not in X21.columns:
            X21[c] = np.nan
    X21 = X21[X.columns]
    for c in CAT_COLS:
        if c in X21.columns:
            X21[c] = X21[c].astype("category")

    p21 = calib.predict_proba(X21)[:, 1]
    out = pd.DataFrame({"id_alumno": d21["id_alumno"].values, "p_abandono": p21})
    out.to_parquet(PATH_PRED, index=False)
    print(f"Predicciones 2021 guardadas en {PATH_PRED}  (n={len(out):,})")
    print(out["p_abandono"].describe())

if __name__ == "__main__":
    main()
