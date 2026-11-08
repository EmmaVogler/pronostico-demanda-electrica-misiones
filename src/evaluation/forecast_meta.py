"""Metadatos de la grilla de forecast para los CSV de predicciones de test.

Por que existe
--------------
``src/analysis/residual_analysis.py::validate_forecast_grid`` necesita saber con
que ``horizon`` y ``stride`` se generaron realmente las ventanas, para poder
distinguir dos casos que son estructuralmente identicos en un CSV:

    A) ventanas contiguas  (stride == horizon)  -> CSV valido
    B) timestamps fabricados (time_index[-n:]) -> CSV invalido

Ambos producen una serie con deltas de 1 hora. Solo el ``stride`` con el que se
corrio el backtesting los separa. Leer ese valor de ``config.yaml`` no alcanza:
una corrida anterior puede haber usado otro stride, y entonces el diagnostico
rechaza un CSV perfectamente valido (falso positivo) o acepta uno fabricado
(falso negativo).

Por eso cada evaluation script escribe un sidecar ``<csv>.meta.json`` junto al
CSV, y el analisis lo lee con prioridad sobre el config.

Formato del sidecar
-------------------
{
  "model": "lstm",
  "horizon": 24,
  "stride": 24,
  "context_window": 168,
  "test_start": "2025-07-21T09:00:00",
  "n_windows": 164,
  "n_predictions": 3936,
  "generated_at": "2026-09-29T12:34:56",
  "generator": "src/evaluation/lstm_evaluation.py",
  "timestamps_source": "actual.time_index"
}
"""

import json
from datetime import datetime
from pathlib import Path

__all__ = ["write_forecast_meta", "read_forecast_meta", "meta_path_for",
           "infer_stride", "backfill_meta"]

def infer_stride(timestamps, horizon):
    """Deduce el stride de una grilla de ventanas a partir de sus timestamps.

    Para una grilla ``historical_forecasts(horizon, stride)`` sobre una serie
    horaria, los deltas son ``1h`` repetidos ``horizon`` veces y despues un salto
    de ``stride - horizon + 1`` horas. El stride queda univocamente determinado:

        * todos los deltas == 1h  -> stride == horizon (ventanas contiguas)
        * deltas == 1 y deltas > 1 -> stride = 1 + (max_delta - 1) + (horizon - 1)

    Devuelve ``(stride, n_windows)`` o ``(None, None)`` si el patron no encaja
    con ninguna grilla conocida, que es la señal de que los timestamps estan
    fabricados.

    Esto NO reemplaza a ``write_forecast_meta``: inferir es reconstruir metadata
    despues del hecho. Sirve para backfillear corridas antiguas; las nuevas
    deben grabarla el evaluation script.
    """
    import numpy as np
    import pandas as pd

    ts = pd.DatetimeIndex(pd.to_datetime(timestamps)).sort_values()
    if len(ts) < 2 or horizon <= 0:
        return None, None

    deltas = np.unique(ts.diff().dropna().total_seconds().to_numpy() / 3600.0)

    if len(deltas) == 1 and deltas[0] == 1.0:

        if len(ts) < horizon:
            return None, None
        return int(horizon), int(len(ts) // horizon)

    if len(deltas) == 2 and 1.0 in deltas:

        gap = float(max(deltas))
        stride = int(round(gap + horizon - 1))
        if stride < horizon:
            return None, None

        if len(ts) < 2 * horizon:
            return None, None
        n_windows = int(np.ceil(len(ts) / horizon))
        return stride, n_windows

    return None, None

def backfill_meta(csv_path, model, horizon):
    """Escribe un sidecar INFERIDO para un CSV generado antes de que existiera
    ``write_forecast_meta``.

    Marca ``"inferred": true`` y ``"timestamps_source": "unverified"`` para que
    quede trazable que la metadata fue reconstruida y no declarada por el
    evaluation script. Si la inferencia falla, no escribe nada y devuelve None.
    """
    import pandas as pd

    try:
        df = pd.read_csv(csv_path)
    except (OSError, ValueError):
        return None
    if "datetime" not in df.columns:
        return None

    stride, n_windows = infer_stride(df["datetime"], horizon)
    if stride is None:
        return None

    path = write_forecast_meta(
        csv_path, model=model, horizon=horizon, stride=stride,
        n_windows=n_windows, n_predictions=len(df),
    )
    meta = read_forecast_meta(csv_path)
    meta["inferred"] = True
    meta["timestamps_source"] = "unverified"
    meta["note"] = (
        "Metadata reconstruida por inferencia del patron de deltas, no "
        "declarada por el evaluation script. Re-correr el evaluation script "
        "para generar un sidecar con timestamps_source=actual.time_index."
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return path

def meta_path_for(csv_path):
    """Ruta del sidecar asociado a un CSV de predicciones."""
    csv_path = Path(csv_path)
    return csv_path.with_suffix(".meta.json")

def write_forecast_meta(csv_path, model, horizon, stride, context_window=None,
                        test_start=None, n_windows=None, n_predictions=None,
                        generator=None):
    """Escribe el sidecar ``<csv>.meta.json``.

    Se llama justo despues de ``df.to_csv(csv_path)``. La clave
    ``timestamps_source`` queda fijada en ``"actual.time_index"`` porque es la
    unica fuente aceptable: reconstruir con ``time_index[-n:]`` es el bug que
    este modulo existe para evitar que vuelva a colarse sin que nadie lo note.
    """
    meta = {
        "model": model,
        "horizon": int(horizon),
        "stride": int(stride),
        "context_window": int(context_window) if context_window is not None else None,
        "test_start": str(test_start) if test_start is not None else None,
        "n_windows": int(n_windows) if n_windows is not None else None,
        "n_predictions": int(n_predictions) if n_predictions is not None else None,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "generator": generator or "src/evaluation/*_evaluation.py",
        "timestamps_source": "actual.time_index",
    }
    path = meta_path_for(csv_path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return path

def read_forecast_meta(csv_path):
    """Lee el sidecar de un CSV. Devuelve ``None`` si no existe o esta corrupto.

    Devolver ``None`` en vez de levantar excepcion es deliberado: el sidecar es
    metadato opcional, y un CSV de corridas antiguas simplemente no lo tiene.
    Quien llame decide que hacer (el driver cae al stride del config).
    """
    path = meta_path_for(csv_path)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(meta, dict):
        return None

    for key in ("horizon", "stride"):
        if not isinstance(meta.get(key), int) or meta[key] <= 0:
            return None
    return meta
