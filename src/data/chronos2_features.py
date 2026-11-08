import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PREHISTORY_PATH = BASE_DIR / "20-12-2022_31-12-2022_temp_hum.json"

def _load_prehistory() -> pd.DataFrame:
    """Mediciones TEMP/HUM previas (2022-12-20 .. 2022-12-31).

    Se usan únicamente para poder calcular los rezagos de temperatura y
    humedad de las primeras jornadas de 2023 (t-1 y t-24) sin descartar
    registros del dataset.
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

input_path = r"data\processed\demandas_horarias_2023_2026[completed].csv"
output_path = r"data\processed\demandas_horarias_2023_2026[chronos2].csv"

df = pd.read_csv(input_path)

df['timestamp'] = pd.to_datetime(df['FECHA']) + pd.to_timedelta(df['Horario'], unit='h')
df = df.sort_values('timestamp').reset_index(drop=True)

prehistory = _load_prehistory()

ts_index = pd.DatetimeIndex(df['timestamp'])
temp_full = pd.concat([prehistory["temp_medida"], pd.Series(df['TEMP'].values, index=ts_index)]).sort_index()
hum_full = pd.concat([prehistory["hum_medida"], pd.Series(df['HUM'].values, index=ts_index)]).sort_index()
temp_full = temp_full[~temp_full.index.duplicated(keep="last")]
hum_full = hum_full[~hum_full.index.duplicated(keep="last")]

df['temp_lag_1h'] = temp_full.shift(1).reindex(ts_index).values
df['temp_lag_24h'] = temp_full.shift(24).reindex(ts_index).values
df['hum_lag_1h'] = hum_full.shift(1).reindex(ts_index).values
df['hum_lag_24h'] = hum_full.shift(24).reindex(ts_index).values

df['target_lag_1'] = df['TOTAL_SIP'].shift(1)
df['target_lag_24'] = df['TOTAL_SIP'].shift(24)
df = df.dropna(subset=['target_lag_24']).reset_index(drop=True)

df['item_id'] = "SIP_TOTAL"

columns_chronos = [
    'timestamp',
    'item_id',
    'TOTAL_SIP',
    'es_laboral',
    'dia_semana',
    'TEMP_F_24',
    'HUM_F_24',
    'temp_lag_1h',
    'temp_lag_24h',
    'hum_lag_1h',
    'hum_lag_24h',
    'target_lag_1',
    'target_lag_24',
]

df_chronos = df[columns_chronos]

nan_cols = df_chronos.isna().sum()
if int(nan_cols.sum()) > 0:
    raise ValueError(
        f"Quedaron NaN en el dataset: {nan_cols[nan_cols > 0].to_dict()}"
    )

df_chronos.to_csv(output_path, index=False)

print(f"Dataset procesado correctamente y guardado en: {output_path}")
print(f"  Shape: {df_chronos.shape}")
print(f"  Rango: {df_chronos['timestamp'].iloc[0]} -> {df_chronos['timestamp'].iloc[-1]}")
print(f"  Columnas: {list(df_chronos.columns)}")