import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
LAG24_SOURCE_PATH = BASE_DIR / "31-12-2022_temp_hum.json"

def _load_lag24_source() -> pd.DataFrame:
    """Mediciones TEMP/HUM de 2022-12-31 (Open-Meteo archive).

    Se usan para resolver el lag de 24 h de la primera jornada del dataset
    (2023-01-01), cuyas observaciones ``t-24`` caen fuera del rango del CSV.
    """
    if not LAG24_SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de lag 24 h: {LAG24_SOURCE_PATH}"
        )
    with open(LAG24_SOURCE_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

    hourly = payload["hourly"]
    times = pd.to_datetime(hourly["time"])
    return pd.DataFrame(
        {
            "FECHA": times.normalize(),
            "Horario": times.hour.astype(int),
            "TEMP": hourly["temperature_2m"],
            "HUM": hourly["relative_humidity_2m"],
        }
    )

def main():

    input_path = Path(
        "data/processed/demandas_horarias_2023_2026[completed].csv"
    )
    output_path = Path(
        "data/processed/demandas_horarias_2023_2026[sarimax].csv"
    )

    df = pd.read_csv(input_path)

    df["FECHA"] = pd.to_datetime(df["FECHA"])
    df["Horario"] = df["Horario"].astype(int)

    df = (
        df.sort_values(["FECHA", "Horario"], ascending=True)
          .reset_index(drop=True)
    )

    hora = df["Horario"]

    dia_anio = df["FECHA"].dt.dayofyear

    anio = df["FECHA"].dt.year

    dias_en_anio = np.where(
        pd.to_datetime(anio.astype(str) + "-12-31").dt.dayofyear == 366,
        366,
        365,
    )

    mes = df["FECHA"].dt.month

    dia_semana = df["FECHA"].dt.dayofweek

    df["hora_sin"] = np.sin(2 * np.pi * hora / 24)
    df["hora_cos"] = np.cos(2 * np.pi * hora / 24)

    df["dia_semana_sin"] = np.sin(2 * np.pi * dia_semana / 7)
    df["dia_semana_cos"] = np.cos(2 * np.pi * dia_semana / 7)

    df["dia_anio_sin"] = np.sin(2 * np.pi * (dia_anio - 1) / dias_en_anio)
    df["dia_anio_cos"] = np.cos(2 * np.pi * (dia_anio - 1) / dias_en_anio)

    df["mes_sin"] = np.sin(2 * np.pi * (mes - 1) / 12)
    df["mes_cos"] = np.cos(2 * np.pi * (mes - 1) / 12)

    measured = df[["FECHA", "Horario", "TEMP", "HUM"]].copy()

    source = _load_lag24_source()
    measured = (
        pd.concat([source, measured], ignore_index=True)
        .drop_duplicates(subset=["FECHA", "Horario"], keep="last")
        .sort_values(["FECHA", "Horario"])
    )

    measured_lag = measured.rename(
        columns={"TEMP": "TEMP_LAG24", "HUM": "HUM_LAG24"}
    )
    measured_lag["FECHA"] = measured_lag["FECHA"] + pd.Timedelta(days=1)

    df = df.merge(measured_lag, on=["FECHA", "Horario"], how="left")

    missing_lag = int(df["TEMP_LAG24"].isna().sum() + df["HUM_LAG24"].isna().sum())
    if missing_lag:
        raise ValueError(
            f"Quedaron {missing_lag} valores NaN en TEMP_LAG24/HUM_LAG24; "
            "revisar la continuidad del dataset o el JSON de 2022-12-31."
        )

    df["target_lag_1"] = df["TOTAL_SIP"].shift(1)
    #el reazgo usa solo valores previos; evita fuga.
    df["target_lag_24"] = df["TOTAL_SIP"].shift(24)
    df = df.dropna(subset=["target_lag_24"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    columnas_finales = [
        "FECHA", "Horario", "TOTAL_SIP", "dia_semana", "es_laboral",
        "TEMP_F_24", "HUM_F_24", "TEMP_LAG24", "HUM_LAG24",
        "hora_sin", "hora_cos",
        "dia_semana_sin", "dia_semana_cos", "dia_anio_sin", "dia_anio_cos",
        "mes_sin", "mes_cos",
        "target_lag_1", "target_lag_24",
    ]
    df = df[columnas_finales]

    df.to_csv(output_path, index=False)

    print(f"Archivo generado: {output_path}")
    print(f"Registros: {len(df):,}")

if __name__ == "__main__":
    main()