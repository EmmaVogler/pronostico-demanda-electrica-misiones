"""
Script para extraer características de humedad y temperatura desde el SMN (Servicio Meteorológico Nacional).

Este módulo descarga datos meteorológicos horarios de Posadas Aero, los procesa y fusiona con 
el dataset principal de demanda eléctrica para incluir variables exógenas climáticas.

Requisitos:
    - requests: Para descargar archivos HTTP con manejo robusto de errores.
    - pandas: Para manipulación de DataFrames.
    - pathlib: Para gestión de rutas de archivos.
"""

import logging
import os
import time
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import pandas as pd
import requests

LOG_FILE = Path(__file__).resolve().parent / "humidity_temperature_features.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(LOG_FILE)),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

SMN_DATA_URL_TEMPLATE = "https://ssl.smn.gob.ar/dpd/descarga_opendata.php?file=observaciones/datohorario{date}.txt"

SCRIPT_DIR = Path(__file__).parent.parent
DATA_RAW_DIR = SCRIPT_DIR.parent / "data" / "raw"
DATA_PROCESSED_DIR = SCRIPT_DIR.parent / "data" / "processed"

DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TEMPORAL_FILE_NAME = "datohorario_ultimo_dia.txt"

DEMANDA_FILE_NAME = "demandas_horarias_2023_2026[exogenas].csv"

OUTPUT_FILE_NAME = "demandas_horarias_2023_2026[exogenas_temp_hum].csv"

ESTACION_POSADAS_AERO = "POSADAS AERO"

OUTPUT_COLUMNS = {
    "FECHA": "datetime",
    "Horario": int,
    "TEMP": float,
    "HUM": float
}

MAX_RETRIES = 3

RETRY_DELAY = 5

def _parse_smn_line(line: str, line_num: int) -> Optional[Dict[str, object]]:
    """
    Parsea una línea del archivo del SMN.

    Formato esperado (ejemplo):
    02072026     0  14.5   97  1019.3  180   11     POSADAS AERO

    Se asume:
      - tokens[0] = FECHA (DDMMYYYY)
      - tokens[1] = HORA (HH)
      - tokens[2] = TEMP
      - tokens[3] = HUM
      - station name: suele estar después de varios campos numéricos (índice 7 en el ejemplo)

    Args:
        line: Línea cruda del archivo.
        line_num: Número de línea para mensajes de error.

    Returns:
        Diccionario con los campos extraídos o None si la línea es inválida.
    """
    try:
        line = line.strip()
        if not line:
            return None

        tokens = line.split()
        if len(tokens) < 5:

            logger.debug(f"Línea {line_num}: Tokens insuficientes: {tokens}")
            return None

        date_str = tokens[0]
        hour_str = tokens[1]

        if not (date_str.isdigit() and (len(date_str) == 8)):
            logger.debug(f"Línea {line_num}: Fecha inválida: {date_str}")
            return None

        if not hour_str.isdigit():
            logger.debug(f"Línea {line_num}: Hora inválida: {hour_str}")
            return None

        try:
            fecha_dt = datetime.strptime(date_str, "%d%m%Y")
            fecha = fecha_dt.date()
        except ValueError as e:
            logger.debug(f"Línea {line_num}: Error parseando fecha '{date_str}': {e}")
            return None

        try:
            horario = int(hour_str)
        except ValueError:
            logger.debug(f"Línea {line_num}: Hora no numérica: {hour_str}")
            return None

        temp = None
        hum = None
        if len(tokens) > 2:
            try:
                temp = float(tokens[2])
            except Exception:
                temp = None
        if len(tokens) > 3:
            try:
                hum = float(tokens[3])
            except Exception:
                hum = None

        if len(tokens) >= 8:
            station_name = " ".join(tokens[7:]).strip()
        else:

            station_name = " ".join(tokens[4:]).strip()

        if not station_name:
            logger.debug(f"Línea {line_num}: No se detectó nombre de estación.")
            return None

        return {
            "NOMBRE": station_name.upper(),
            "FECHA": fecha,
            "Horario": int(horario),
            "TEMP": temp,
            "HUM": hum
        }

    except Exception as e:
        logger.error(f"Línea {line_num}: Error inesperado al parsear. {e}")
        return None

def _filter_by_station(records, station_name: str) -> Tuple[pd.DataFrame, int]:
    """
    Filtra los registros por nombre de estación meteorológica.

    Acepta tanto una lista de diccionarios (registros) como un DataFrame.

    Returns:
        Tupla con (DataFrame filtrado, cantidad descartada).
    """

    station_name_norm = station_name.strip().upper()

    if isinstance(records, pd.DataFrame):
        df = records.copy()
    else:

        df = pd.DataFrame(records)

    if df.empty:
        return pd.DataFrame(), 0

    if "NOMBRE" not in df.columns:
        logger.warning("No se encontró la columna 'NOMBRE' en los registros. No se podrá filtrar por estación.")
        return pd.DataFrame(), len(df)

    df["NOMBRE"] = df["NOMBRE"].astype(str).str.strip().str.upper()
    filtered_df = df[df["NOMBRE"] == station_name_norm].copy()
    discarded = len(df) - len(filtered_df)

    if discarded > 0:
        logger.info(f"Estación '{station_name}': {discarded} registros descartados (no corresponden).")

    return filtered_df, int(discarded)

def _validate_and_deduplicate_smn_data(
    df: pd.DataFrame, 
    station_name: str
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Valida y deduplica los datos del SMN.

    Asume que `df` ya está filtrado por estación (o contiene sólo la estación de interés).

    Returns:
        Tupla con (DataFrame filtrado, estadísticas de validación).
    """
    stats = {
        "total_records": len(df),
        "station_records": len(df),
        "duplicates_removed": 0,
        "missing_temp": 0,
        "missing_hum": 0
    }

    if df.empty:
        logger.warning(f"No se proporcionaron registros para la estación '{station_name}'.")
        return pd.DataFrame(), stats

    required_columns = ["FECHA", "Horario", "TEMP", "HUM"]
    missing_cols = [col for col in required_columns if col not in df.columns]

    if missing_cols:
        logger.error(f"Faltan columnas requeridas en el archivo del SMN: {missing_cols}")
        logger.error("No se puede continuar con el procesamiento.")
        raise ValueError(f"Columnas faltantes: {missing_cols}")

    duplicate_count = df.duplicated(subset=["FECHA", "Horario"], keep="first").sum()
    stats["duplicates_removed"] = int(duplicate_count)

    if duplicate_count > 0:
        logger.warning(f"Se encontraron {duplicate_count} registros duplicados (FECHA, Horario). "
                       f"Se conservó el primer registro y se descartaron los demás.")

    df = df.drop_duplicates(subset=["FECHA", "Horario"], keep="first")

    stats["missing_temp"] = int(df["TEMP"].isna().sum())
    stats["missing_hum"] = int(df["HUM"].isna().sum())

    if stats["missing_temp"] > 0:
        logger.warning(f"Valores faltantes en TEMP: {stats['missing_temp']}")

    if stats["missing_hum"] > 0:
        logger.warning(f"Valores faltantes en HUM: {stats['missing_hum']}")

    return df, stats

def _download_smn_data(
    url: str, 
    output_path: Path,
    max_retries: int = MAX_RETRIES,
    retry_delay: float = RETRY_DELAY
) -> bool:
    """
    Descarga el archivo del SMN con manejo de errores y reintentos.
    
    Args:
        url: URL del archivo a descargar.
        output_path: Ruta donde guardar el archivo temporal.
        max_retries: Número máximo de reintentos.
        retry_delay: Tiempo de espera entre reintentos (segundos).
    
    Returns:
        True si la descarga fue exitosa, False en caso de error.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    if output_path.exists() and output_path.stat().st_size > 0:
        logger.info(f"Archivo ya existe, salto descarga: {output_path.name}")
        return True

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Intento {attempt}/{max_retries} de descarga desde {url}")
            
            response = requests.get(url, headers=headers, timeout=30)
            

            if response.status_code == 200:

                content_text = response.content.decode("latin-1", errors="ignore")
                normalized = content_text.lower()
                if "el archivo no existe" in normalized or "no existe" in normalized:
                    logger.warning(f"El servidor devolvió un mensaje de error: {content_text[:200]}")

                    return False
                    
                with open(output_path, "wb") as f:
                    f.write(response.content)
                    
                logger.info(f"Archivo descargado exitosamente: {output_path.name}")
                return True
            else:
                logger.warning(f"Código HTTP {response.status_code}. Reintentando en {retry_delay}s...")
                if attempt < max_retries:
                    time.sleep(retry_delay)
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error en la descarga (Intento {attempt}): {e}")
            if attempt < max_retries:
                time.sleep(retry_delay)
    
    logger.error(f"Falló la descarga después de {max_retries} intentos.")
    return False

def _load_demand_data(file_path: Path) -> pd.DataFrame:
    """
    Carga el dataset principal de demanda eléctrica y normaliza FECHA y Horario para el merge.

    Args:
        file_path: Ruta del archivo CSV con demandas.

    Returns:
        DataFrame con los datos de demanda.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Archivo de demanda no encontrado: {file_path}")

    logger.info(f"Cargando dataset de demanda desde {file_path.name}")
    df_demand = pd.read_csv(file_path)

    date_cols = ["FECHA", "Horario"]
    if not all(col in df_demand.columns for col in date_cols):
        available_cols = list(df_demand.columns)
        raise ValueError(f"Columnas requeridas '{', '.join(date_cols)}' no encontradas. "
                         f"Columnas disponibles: {available_cols}")

    df_demand["FECHA"] = pd.to_datetime(df_demand["FECHA"], format="%Y-%m-%d", errors="coerce").dt.date
    invalid_dates = df_demand["FECHA"].isna().sum()
    if invalid_dates > 0:
        logger.warning(f"Se encontraron {invalid_dates} filas con FECHA inválida en el dataset de demanda.")

    def _parse_horario(value):
        if pd.isna(value):
            return None
        text = str(value).strip()
        if not text:
            return None
        if ":" in text:
            parts = text.split(":")
            if parts[0].isdigit():
                return int(parts[0])
            return None
        if text.isdigit():
            return int(text)
        try:
            numeric = float(text)
            return int(numeric)
        except Exception:
            return None

    df_demand["Horario"] = df_demand["Horario"].map(_parse_horario).astype("Int64")
    invalid_horario = df_demand["Horario"].isna().sum()
    if invalid_horario > 0:
        logger.warning(f"Se encontraron {invalid_horario} filas con Horario inválido en el dataset de demanda.")

    return df_demand

def _merge_climate_features(
    demand_df: pd.DataFrame, 
    climate_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Fusiona las características climáticas con el dataset de demanda mediante left merge.
    
    Args:
        demand_df: DataFrame principal con demandas.
        climate_df: DataFrame con características climáticas (TEMP, HUM).
    
    Returns:
        DataFrame fusionado con las columnas TEMP y HUM agregadas.
    """
    logger.info("Realizando left merge para agregar características climáticas...")
    
    df_merged = demand_df.merge(
        climate_df,
        on=["FECHA", "Horario"],
        how="left"
    )
    
    return df_merged

def _save_processed_data(df: pd.DataFrame, output_path: Path) -> None:
    """
    Guarda el DataFrame procesado en un archivo CSV.
    
    Args:
        df: DataFrame a guardar.
        output_path: Ruta del archivo de salida.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    

    df.to_csv(output_path, index=False)
    
    logger.info(f"Datos guardados exitosamente en: {output_path.name}")

def _cleanup_temporal_file(file_path: Path) -> None:
    """
    Elimina un archivo temporal del directorio de raw.

    Seguridad: Solo elimina archivos que están dentro de DATA_RAW_DIR y que tengan
    extensiones típicas de archivos temporales del SMN (.txt, .dat). Esto evita borrar
    accidentalmente CSVs u otros archivos importantes.

    Args:
        file_path: Ruta del archivo a eliminar.
    """
    try:
        if not file_path.exists():
            return

        resolved = file_path.resolve()
        raw_resolved = DATA_RAW_DIR.resolve()

        is_in_raw = (raw_resolved == resolved.parent) or (raw_resolved in resolved.parents)

        if not is_in_raw:
            logger.warning(f"Se evitó eliminar archivo fuera de data/raw: {resolved}")
            return

        if resolved.suffix.lower() not in (".txt", ".dat"):
            logger.warning(f"Se evitó eliminar archivo con extensión no esperada: {resolved}")
            return

        resolved.unlink()
        logger.info(f"Archivo temporal eliminado: {resolved.name}")

    except Exception as e:
        logger.warning(f"No se pudo eliminar el archivo temporal: {e}")

def _inspect_temporal_file(file_path: Path) -> None:
    """
    Inspecciona el archivo temporal para depuración.
    
    Args:
        file_path: Ruta del archivo a inspeccionar.
    """
    if file_path.exists():
        logger.info(f"Contenido del archivo temporal (primeras 20 líneas):")
        with open(file_path, "r", encoding="latin-1") as f:
            for i, line in enumerate(f):
                if i >= 20:
                    break
                logger.info(f"Línea {i+1}: {repr(line)}")

def extract_humidity_temperature_features() -> pd.DataFrame:
    """
    Función principal que orquesta todo el proceso de extracción y fusión de características.
    
    Proceso:
        1. Descargar archivo del SMN (con reintentos).
        2. Parsear el archivo de ancho fijo.
        3. Filtrar por estación "POSADAS AERO".
        4. Validar y deduplicar datos.
        5. Cargar dataset de demanda principal.
        6. Fusionar características climáticas (TEMP, HUM).
        7. Guardar resultado en archivo CSV.
        8. Eliminar archivo temporal.
    
    Returns:
        DataFrame fusionado con características climáticas agregadas.
    
    Raises:
        ValueError: Si faltan columnas requeridas o ocurren errores de validación.
        FileNotFoundError: Si el archivo de demanda no existe.
    """

    logger.info("=" * 80)
    logger.info("INICIO DEL PROCESO DE EXTRACCIÓN DE CARACTERÍSTICAS CLIMÁTICAS")
    logger.info("=" * 80)

    demand_file = DATA_PROCESSED_DIR / DEMANDA_FILE_NAME
    demand_df = _load_demand_data(demand_file)
    logger.info(f"Dataset de demanda: {len(demand_df)} filas, {len(demand_df.columns)} columnas")

    unique_dates = sorted(d for d in demand_df['FECHA'].unique() if pd.notna(d))

    if not unique_dates:
        raise ValueError("No se encontraron fechas válidas en el dataset de demanda para descargar datos SMN.")

    records = []
    downloaded_files = []

    for dt in unique_dates:

        if isinstance(dt, datetime):
            dt_obj = dt.date()
        elif isinstance(dt, date):
            dt_obj = dt
        else:

            try:
                dt_obj = pd.to_datetime(dt, dayfirst=True).date()
            except Exception:
                logger.warning(f"Fecha inválida en demand_df: {dt}. Se omite.")
                continue

        date_str = dt_obj.strftime("%Y%m%d")
        url = SMN_DATA_URL_TEMPLATE.format(date=date_str)
        temporal_file = DATA_RAW_DIR / f"datohorario{date_str}.txt"

        success = _download_smn_data(url, temporal_file)
        if not success:
            logger.warning(f"No se pudo descargar datos SMN para la fecha {dt_obj} (URL: {url}). Se continúa con otras fechas.")
            continue

        downloaded_files.append(temporal_file)

        logger.info(f"Parseando archivo: {temporal_file.name}")
        with open(temporal_file, "r", encoding="latin-1") as f:
            for line_num, line in enumerate(f, start=1):
                parsed = _parse_smn_line(line, line_num)
                if parsed:
                    records.append(parsed)

    if not records:
        logger.error("No se pudieron parsear registros válidos de los archivos SMN descargados.")

        if downloaded_files:
            _inspect_temporal_file(downloaded_files[-1])
        raise ValueError("No se encontraron datos válidos en los archivos del SMN.")

    logger.info(f"Registros parseados exitosamente en total: {len(records)}")

    df_raw = pd.DataFrame(records)

    station_df, discarded = _filter_by_station(df_raw, ESTACION_POSADAS_AERO)

    if not station_df.empty:
        logger.info(f"Registros de '{ESTACION_POSADAS_AERO}': {len(station_df)}")

    station_df, stats = _validate_and_deduplicate_smn_data(station_df, ESTACION_POSADAS_AERO)

    if not station_df.empty:
        logger.info(f"Estadísticas de validación:")
        logger.info(f"  - Registros totales del SMN: {stats['total_records']}")
        logger.info(f"  - Registros de '{ESTACION_POSADAS_AERO}': {stats['station_records']}")
        logger.info(f"  - Duplicados eliminados: {stats['duplicates_removed']}")
        logger.info(f"  - Valores faltantes en TEMP: {stats['missing_temp']}")
        logger.info(f"  - Valores faltantes en HUM: {stats['missing_hum']}")

    if station_df.empty:
        logger.warning("No hay datos de la estación 'POSADAS AERO' para procesar.")

        return demand_df

    climate_df = station_df[["FECHA", "Horario", "TEMP", "HUM"]].copy()

    climate_df["FECHA"] = pd.to_datetime(climate_df["FECHA"], errors="coerce").dt.date
    climate_df["Horario"] = pd.to_numeric(climate_df["Horario"], errors="coerce").astype("Int64")
    invalid_climate_hours = climate_df["Horario"].isna().sum()
    if invalid_climate_hours > 0:
        logger.warning(f"Se encontraron {invalid_climate_hours} filas del SMN con Horario inválido.")

    df_merged = _merge_climate_features(demand_df, climate_df)

    logger.info(f"Dataset fusionado: {len(df_merged)} filas")

    output_file = DATA_PROCESSED_DIR / OUTPUT_FILE_NAME
    _save_processed_data(df_merged, output_file)

    for fpath in downloaded_files:
        _cleanup_temporal_file(fpath)

    logger.info("=" * 80)
    logger.info("ESTADÍSTICAS FINALES")
    logger.info("=" * 80)
    logger.info(f"Registros descargados del SMN (parseados): {len(df_raw)}")
    logger.info(f"Registros de '{ESTACION_POSADAS_AERO}': {stats.get('station_records', 0)}")
    logger.info(f"Filas fusionadas en el dataset final: {len(df_merged)}")
    logger.info(f"Valores faltantes en TEMP (después del merge): {df_merged['TEMP'].isna().sum()}")
    logger.info(f"Valores faltantes en HUM (después del merge): {df_merged['HUM'].isna().sum()}")

    return df_merged

def main() -> None:
    """
    Punto de entrada del script. Ejecuta la extracción de características climáticas.
    
    Uso:
        python humidity_temperature_features.py
    
    Salida:
        - Archivo CSV en: code/data/processed/demandas_horarias_2023_2026[exogenas_temp_hum].csv
        - Log file: code/src/data/humidity_temperature_features.log
    """
    logger.info("Iniciando script humidity_temperature_features.py")
    
    try:
        result_df = extract_humidity_temperature_features()
        

        print("\n" + "=" * 80)
        print("RESULTADO FINAL (primeras 10 filas):")
        print("=" * 80)
        print(result_df.head(10).to_string())
        print("\n" + "=" * 80)
        print(f"Total de filas procesadas: {len(result_df)}")
        print("=" * 80 + "\n")
        
    except Exception as e:
        logger.error(f"El script terminó con error: {e}")
        raise

if __name__ == "__main__":
    main()
