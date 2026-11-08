import pandas as pd
import requests
import time
import logging
import os
from datetime import datetime, date

INPUT_FILE = os.path.join('data', 'processed', 'demandas_horarias_2023_2026[exogenas_temp_hum].csv')
OUTPUT_FILE = os.path.join('data', 'processed', 'demandas_horarias_2023_2026[exogenas_temp_hum_completed].csv')
LOG_FILE = os.path.join('src', 'data', 'humidity_temperature_features.log')

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def fetch_weather_data(lat, lon, start_date, end_date):
    """
    Consulta la API de Open-Meteo. 
    Usa el endpoint de archive para datos históricos (más de 2 días de antigüedad)
    y el de forecast para datos recientes o futuros.
    """
    target_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    today = date.today()
    

    if target_date < today:
        url = "https://archive-api.open-meteo.com/v1/archive"
    else:
        url = "https://api.open-meteo.com/v1/forecast"
        
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m",
        "timezone": "auto",
        "start_date": start_date,
        "end_date": end_date
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"Error consultando API para {start_date}: {e}")
        return None

def main():
    logging.info("--- Iniciando proceso de completado de datos de temperatura y humedad ---")
    

    if not os.path.exists(INPUT_FILE):
        logging.error(f"Archivo de entrada no encontrado: {os.path.abspath(INPUT_FILE)}")
        return

    df = pd.read_csv(INPUT_FILE)
    

    missing_mask = df['TEMP'].isna() | df['HUM'].isna()
    missing_count_initial = missing_mask.sum()
    
    if missing_count_initial == 0:
        logging.info("No se encontraron registros faltantes. El archivo ya está completo.")

        df.to_csv(OUTPUT_FILE, index=False)
        return

    missing_dates = sorted(df.loc[missing_mask, 'FECHA'].unique())
    logging.info(f"Se encontraron {missing_count_initial} registros faltantes distribuidos en {len(missing_dates)} fechas.")
    

    lat, lon = -27.3918, -55.9238
    
    processed_count = 0
    for d in missing_dates:
        logging.info(f"Procesando fecha {d} ({processed_count + 1}/{len(missing_dates)})...")
        
        weather_data = fetch_weather_data(lat, lon, d, d)
        
        if weather_data and 'hourly' in weather_data:
            hourly = weather_data['hourly']
            times = hourly.get('time', [])
            temps = hourly.get('temperature_2m', [])
            hums = hourly.get('relative_humidity_2m', [])
            
            for i in range(len(times)):
                try:

                    dt = datetime.strptime(times[i], '%Y-%m-%dT%H:%M')
                    hour = dt.hour
                    

                    idx_mask = (df['FECHA'] == d) & (df['Horario'] == hour)
                    
                    if idx_mask.any():
                        if pd.isna(df.loc[idx_mask, 'TEMP']).any():
                            df.loc[idx_mask, 'TEMP'] = temps[i]
                        if pd.isna(df.loc[idx_mask, 'HUM']).any():
                            df.loc[idx_mask, 'HUM'] = hums[i]
                except Exception as ex:
                    logging.warning(f"Error procesando hora {times[i]} para la fecha {d}: {ex}")
        
        processed_count += 1

        time.sleep(0.5)
        

    final_missing_mask = df['TEMP'].isna() | df['HUM'].isna()
    remaining_count = final_missing_mask.sum()
    
    result_msg = f"Comprobación final: Quedaron {remaining_count} registros sin rellenar."
    logging.info(result_msg)
    print(f"\n{result_msg}")
    

    try:
        df.to_csv(OUTPUT_FILE, index=False)
        logging.info(f"Nuevo archivo guardado exitosamente en: {OUTPUT_FILE}")
    except Exception as e:
        logging.error(f"Error al guardar el archivo de salida: {e}")

if __name__ == "__main__":
    main()
 