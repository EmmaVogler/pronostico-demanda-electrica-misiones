import os
import time
import logging
import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import statsmodels.api as sm

INPUT_FILE = os.path.join('data', 'processed', 'demandas_horarias_2023_2026[exogenas_completed_ordered].csv')
OUTPUT_FILE = os.path.join('data', 'processed', 'demandas_horarias_2023_2026[completed].csv')
LOG_FILE = os.path.join('src', 'data', 'humidity_temperature_features.log')

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def create_robust_session():
    """
    Crea una sesión HTTP con estrategia de reintentos exponenciales
    y headers de navegador real para evitar bloqueos SSL (443) o Rate Limits (429).
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=2,  
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
        "Accept-Language": "es-AR,es;q=0.9,en;q=0.8"
    })
    return session

def fetch_forecast_chunk(session, lat, lon, start_date, end_date):
    """
    Consulta la Previous Runs API de Open-Meteo para un rango acotado.
    """
    url = "https://previous-runs-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m_previous_day1,relative_humidity_2m_previous_day1",
        "timezone": "America/Argentina/Buenos_Aires"
    }
    
    try:
        response = session.get(url, params=params, timeout=20)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"Error consultando API para el periodo {start_date} a {end_date}: {e}")
        return None

def impute_missing_ucm(series, col_name):
    """
    Imputa valores nulos en una serie temporal mediante UnobservedComponents
    (Componentes No Observables) utilizando el suavizado del filtro de Kalman.
    """
    missing_count = series.isna().sum()
    if missing_count == 0:
        logging.info(f"No hay valores faltantes en {col_name}.")
        return series

    logging.info(f"Imputando {missing_count} valores faltantes en '{col_name}' con UnobservedComponents (seasonal=24)...")

    series_clean = series.copy().astype(float)

    try:

        model = sm.tsa.UnobservedComponents(
            series_clean,
            level='local level',
            seasonal=24
        )
        

        res = model.fit(disp=False, maxiter=200)
        

        predictions = res.get_prediction()
        smoothed_means = predictions.predicted_mean
        

        series_imputed = series_clean.fillna(smoothed_means)
        logging.info(f"Imputación UCM completada exitosamente para '{col_name}'.")
        return series_imputed

    except Exception as e:
        logging.error(f"Error en el ajuste de UnobservedComponents para {col_name}: {e}. Se utilizará interpolación como fallback.")
        return series_clean.interpolate(method='time').bfill().ffill()

def main():
    logging.info("=== Iniciando procesamiento e integración de pronósticos Day-Ahead ===")
    
    if not os.path.exists(INPUT_FILE):
        logging.error(f"Archivo no encontrado: {os.path.abspath(INPUT_FILE)}")
        return

    df = pd.read_csv(INPUT_FILE)
    logging.info(f"Dataset cargado correctamente: {len(df)} registros.")

    lat, lon = -27.3918, -55.9238

    date_blocks = pd.date_range(start='2023-01-01', end='2026-01-01', freq='MS')
    
    session = create_robust_session()
    forecast_chunks = []

    logging.info("Descargando datos meteorológicos de Open-Meteo en bloques mensuales con delais...")
    
    for i in range(len(date_blocks) - 1):
        start_str = date_blocks[i].strftime('%Y-%m-%d')

        end_date = date_blocks[i+1] - pd.Timedelta(days=1)
        end_str = end_date.strftime('%Y-%m-%d')

        logging.info(f" -> Pidiendo bloque [{i+1}/{len(date_blocks)-1}]: {start_str} al {end_str}")
        
        data = fetch_forecast_chunk(session, lat, lon, start_str, end_str)
        
        if data and 'hourly' in data:
            hourly = data['hourly']
            chunk_df = pd.DataFrame({
                'datetime': pd.to_datetime(hourly['time']),
                'TEMP_F_24': hourly.get('temperature_2m_previous_day1'),
                'HUM_F_24': hourly.get('relative_humidity_2m_previous_day1')
            })
            forecast_chunks.append(chunk_df)
        else:
            logging.warning(f"No se pudieron obtener datos para el periodo {start_str} a {end_str}")

        time.sleep(3.5)

    if not forecast_chunks:
        logging.error("No se descargó ningún bloque de datos de pronóstico. Abortando.")
        return

    df_forecast_all = pd.concat(forecast_chunks, ignore_index=True)
    df_forecast_all['FECHA'] = df_forecast_all['datetime'].dt.strftime('%Y-%m-%d')
    df_forecast_all['Horario'] = df_forecast_all['datetime'].dt.hour
    df_forecast_all.drop(columns=['datetime'], inplace=True)

    df_forecast_all.drop_duplicates(subset=['FECHA', 'Horario'], inplace=True)

    df['Horario'] = df['Horario'].astype(int)
    

    for col in ['TEMP_F_24', 'HUM_F_24']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    logging.info("Uniendo pronósticos al dataset base...")
    df = pd.merge(df, df_forecast_all, on=['FECHA', 'Horario'], how='left')

    logging.info("Filtrando ceros y valores físicamente imposibles en pronósticos de acá")
    df.loc[df['HUM_F_24'] <= 5.0, 'HUM_F_24'] = np.nan
    df.loc[df['TEMP_F_24'] <= 0.0, 'TEMP_F_24'] = np.nan

    df['TEMP_F_24'] = impute_missing_ucm(df['TEMP_F_24'], 'TEMP_F_24').round(1)
    df['HUM_F_24'] = impute_missing_ucm(df['HUM_F_24'], 'HUM_F_24').round(1)

    columnas_finales = [
        'FECHA', 'Horario', 'TOTAL_SIP', 'dia_semana', 
        'es_laboral', 'TEMP', 'HUM', 'TEMP_F_24', 'HUM_F_24'
    ]
    

    df_resultado = df[columnas_finales]

    try:
        df_resultado.to_csv(OUTPUT_FILE, index=False)
        logging.info(f"=== Proceso finalizado con éxito ===")
        logging.info(f"Archivo exportado en: {os.path.abspath(OUTPUT_FILE)}")
    except Exception as e:
        logging.error(f"Error al escribir el archivo de salida: {e}")

if __name__ == "__main__":
    main()