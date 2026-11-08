import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PREHISTORY_PATH = BASE_DIR / "20-12-2022_31-12-2022_temp_hum.json"

def _load_prehistory() -> pd.DataFrame:
    """Mediciones TEMP/HUM previas (2022-12-20 .. 2022-12-31).

    Se usan únicamente para poder calcular los rezagos de temperatura y
    humedad de las primeras jornadas de 2023 (t-1, t-24 y t-168). La demanda
    no existe antes de 2023-01-01, así que este JSON es la única fuente para
    completar el warm-up meteorológico sin descartar registros.
    """
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

target = df["TOTAL_SIP"]

prehistory = _load_prehistory()

temp_full = pd.concat([prehistory["temp_medida"], df["TEMP"]]).sort_index()
hum_full = pd.concat([prehistory["hum_medida"], df["HUM"]]).sort_index()
temp_full = temp_full[~temp_full.index.duplicated(keep="last")]
hum_full = hum_full[~hum_full.index.duplicated(keep="last")]

df["temp_lag_1h"] = temp_full.shift(1).reindex(df.index)
df["temp_lag_24h"] = temp_full.shift(24).reindex(df.index)

df["hum_lag_1h"] = hum_full.shift(1).reindex(df.index)
df["hum_lag_24h"] = hum_full.shift(24).reindex(df.index)

df["temperatura"] = df["TEMP_F_24"]
df["humedad"] = df["HUM_F_24"]

df["rolling_mean_24h"] = target.rolling(window=24, min_periods=1).mean()

df["rolling_std_24h"] = (
    target.rolling(window=24, min_periods=1).std().fillna(0.0)
)
df["rolling_mean_temp_24h"] = (
    temp_full.rolling(window=24, min_periods=1).mean().reindex(df.index)
)

hora = df.index.hour
df["hora_sin"] = np.sin(2 * np.pi * hora / 24)
df["hora_cos"] = np.cos(2 * np.pi * hora / 24)

dia_semana = df.index.dayofweek
df["dia_sem_sin"] = np.sin(2 * np.pi * dia_semana / 7)
df["dia_sem_cos"] = np.cos(2 * np.pi * dia_semana / 7)

df["es_feriado"] = 1 - df["es_laboral"]

df["indice_temporal"] = np.arange(len(df))

t_hours = (df.index - df.index[0]).total_seconds() / 3600.0

fourier_periods = {"daily": 24.0, "weekly": 168.0, "yearly": 8765.82}
k_daily, k_weekly, k_yearly = 6, 4, 2

for period_name, period in fourier_periods.items():
    k = {"daily": k_daily, "weekly": k_weekly, "yearly": k_yearly}[period_name]
    for i in range(1, k + 1):
        df[f"fourier_{period_name}_sin_{i}"] = np.sin(2 * np.pi * i * t_hours / period)
        df[f"fourier_{period_name}_cos_{i}"] = np.cos(2 * np.pi * i * t_hours / period)

fourier_cols = [
    f"fourier_{pn}_sin_{i}" for pn, k in [("daily", k_daily), ("weekly", k_weekly), ("yearly", k_yearly)] for i in range(1, k + 1)
] + [
    f"fourier_{pn}_cos_{i}" for pn, k in [("daily", k_daily), ("weekly", k_weekly), ("yearly", k_yearly)] for i in range(1, k + 1)
]

covariates_cols = [
    "FECHA", "Horario",
    "temperatura", "humedad",
    "temp_lag_1h", "temp_lag_24h",
    "hum_lag_1h", "hum_lag_24h",
    "rolling_mean_24h", "rolling_std_24h", "rolling_mean_temp_24h",
    "hora_sin", "hora_cos",
    "dia_sem_sin", "dia_sem_cos",
    "es_feriado",
    "indice_temporal",
] + fourier_cols

df_final = df[["TOTAL_SIP"] + covariates_cols].copy()

nan_cols = df_final.isna().sum()
if int(nan_cols.sum()) > 0:
    raise ValueError(
        f"Quedaron NaN en el dataset: {nan_cols[nan_cols > 0].to_dict()}"
    )

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "demandas_horarias_2023_2026[lightgbm].csv"
df_final.to_csv(output_path)

print(f"✓ Dataset con covariables guardado en: {output_path}")
print(f"  Shape: {df_final.shape}")
print(f"  Rango: {df_final.index[0]} -> {df_final.index[-1]}")
print(f"  Columnas: {list(df_final.columns)}")
print(f"\nPrimeras 5 filas:")
print(df_final.head())
print(f"\nÚltimas 5 filas:")
print(df_final.tail())
print(f"\nResumen (NaN por columna):")
print(df_final.isna().sum())
