"""Genera el dataset [lstm].csv con rezagos de temperatura/humedad.

Diseño para RNNModel("LSTM"):
- Solo usa future_covariates (no past_covariates).
- Los rezagos de TEMP/HUM (lag_1h, lag_24h) se materializan aquí y se pasan
  como future_covariates. RNNModel los usa en cada paso del horizonte:
    - Para predecir t+k, usa future_cov[t+k] = TEMP(t+k-lag).
    - Con lag=24: TEMP(t+k-24) siempre es pasado o presente → sin leakage.
    - Con lag=1:  TEMP(t+k-1) es pasado para k≥2; para k=1 es contemporáneo
      (TEMP(t)), pero el RNN lo procesa vía su memoria interna.
- La serie extiende hacia atrás con el JSON 20-12-2022_31-12-2022 para
  completar el warm-up de los rezagos de las primeras jornadas de 2023.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PREHISTORY_PATH = BASE_DIR / "20-12-2022_31-12-2022_temp_hum.json"

def _load_prehistory() -> pd.DataFrame:
    """Mediciones TEMP/HUM previas (2022-12-20 .. 2022-12-31)."""
    if not PREHISTORY_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de prehistórico meteorológico: {PREHISTORY_PATH}"
        )
    with open(PREHISTORY_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

    hourly = payload["hourly"]
    times = pd.to_datetime(hourly["time"])
    return (
        pd.DataFrame(
            {
                "datetime": times,
                "temp_medida": hourly["temperature_2m"],
                "hum_medida": hourly["relative_humidity_2m"],
            }
        )
        .set_index("datetime")
        .sort_index()
    )

df = pd.read_csv("data/processed/demandas_horarias_2023_2026[completed].csv")
df["FECHA"] = pd.to_datetime(df["FECHA"])
df["Horario"] = pd.to_numeric(df["Horario"])
df["datetime"] = df["FECHA"] + pd.to_timedelta(df["Horario"], unit="h")
df = df.set_index("datetime").sort_index()

prehistory = _load_prehistory()

temp_full = pd.concat([prehistory["temp_medida"], df["TEMP"]]).sort_index()
hum_full = pd.concat([prehistory["hum_medida"], df["HUM"]]).sort_index()
temp_full = temp_full[~temp_full.index.duplicated(keep="last")]
hum_full = hum_full[~hum_full.index.duplicated(keep="last")]

df["temp_lag_1h"] = temp_full.shift(1).reindex(df.index)
df["temp_lag_24h"] = temp_full.shift(24).reindex(df.index)
df["hum_lag_1h"] = hum_full.shift(1).reindex(df.index)
df["hum_lag_24h"] = hum_full.shift(24).reindex(df.index)

df["target_lag_1"] = df["TOTAL_SIP"].shift(1)
df["target_lag_24"] = df["TOTAL_SIP"].shift(24)
df = df.dropna(subset=["target_lag_24"])

df["temperatura"] = df["TEMP_F_24"]
df["humedad"] = df["HUM_F_24"]

hora = df.index.hour
df["hora_sin"] = np.sin(2 * np.pi * hora / 24)
df["hora_cos"] = np.cos(2 * np.pi * hora / 24)

dia_anio = df.index.dayofyear
anio = df.index.year
dias_en_anio = np.where(
    pd.to_datetime(anio.astype(str) + "-12-31").dayofyear == 366,
    366, 365,
)
df["dia_anio_sin"] = np.sin(2 * np.pi * (dia_anio - 1) / dias_en_anio)
df["dia_anio_cos"] = np.cos(2 * np.pi * (dia_anio - 1) / dias_en_anio)

df["es_feriado"] = 1 - df["es_laboral"]

covariates_cols = [
    "FECHA", "Horario",
    "temperatura", "humedad",
    "temp_lag_1h", "temp_lag_24h",
    "hum_lag_1h", "hum_lag_24h",
    "target_lag_1", "target_lag_24",
    "es_feriado",
    "hora_sin", "hora_cos",
    "dia_anio_sin", "dia_anio_cos",
]

df_final = df[["TOTAL_SIP"] + covariates_cols].copy()

nan_cols = df_final.isna().sum()
if int(nan_cols.sum()) > 0:
    raise ValueError(
        f"Quedaron NaN en el dataset: {nan_cols[nan_cols > 0].to_dict()}"
    )

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "demandas_horarias_2023_2026[lstm].csv"
df_final.to_csv(output_path)

print(f"✓ Dataset LSTM guardado en: {output_path}")
print(f"  Shape: {df_final.shape}")
print(f"  Rango: {df_final.index[0]} -> {df_final.index[-1]}")
print(f"  Columnas: {list(df_final.columns)}")
print(f"\nPrimeras 5 filas:")
print(df_final.head())
print(f"\nÚltimas 5 filas:")
print(df_final.tail())
print(f"\nResumen (NaN por columna):")
print(df_final.isna().sum())
