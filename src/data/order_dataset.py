import pandas as pd

INPUT_FILE = "data/processed/demandas_horarias_2023_2026[exogenas_temp_hum_completed].csv"
OUTPUT_FILE = "data/processed/demandas_horarias_2023_2026[exogenas_completed_ordered].csv"

df = pd.read_csv(INPUT_FILE)

df["FECHA"] = pd.to_datetime(df["FECHA"])
df = df.sort_values(["FECHA", "Horario"]).reset_index(drop=True)

df["FECHA"] = df["FECHA"].dt.strftime("%Y-%m-%d")

df.to_csv(OUTPUT_FILE, index=False)

print(f"Dataset ordenado guardado en: {OUTPUT_FILE}")