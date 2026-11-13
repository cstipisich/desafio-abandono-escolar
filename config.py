from pathlib import Path

BASE = Path(r"D:\Ciencia_Datos_UNO\Aplicacion_Ciencia_Datos\Clase_6")
SOL  = BASE / "solucion"

PATH_HIST    = BASE / "chile_2014_2020.csv"
PATH_2021    = BASE / "chile_2021.csv"
PATH_MODELO  = SOL / "modelo_riesgo.pkl"
PATH_PRED    = SOL / "pred_2021.parquet"
PATH_ENTREGA = SOL / "entrega.csv"

COLS_ETIQUETA = "abandono_t1"

BUDGET   = 1_000_000.0
K_EFECTO = 0.6
TAU      = 200.0

ANIOS_TRAIN = [2019, 2020]

CAT_COLS = [
    "cod_reg_rbd", "cod_depe", "cod_depe2", "cod_ense", "cod_ense2",
    "cod_grado", "cod_jor", "cod_tip_cur", "cod_rama", "cod_espe",
    "gen_alu", "cod_reg_alu", "sit_fin",
]

NUM_COLS = [
    "rural_rbd", "edad_alu", "prom_gral", "asistencia", "n_reg",
    "cambio_esc_anio", "egresa", "esc_n", "esc_prom", "esc_asist",
    "esc_rep", "esc_ab_lag", "edad",
]
