import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import gc
import time
import logging
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import torch
torch.backends.fp32_precision = "ieee"
from ruamel.yaml import YAML

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast_meta import write_forecast_meta  
from darts import TimeSeries
from darts.models import RNNModel
from darts.dataprocessing.transformers import Scaler

torch.set_num_threads(1)
warnings.filterwarnings("ignore")

CONFIG_PATH = Path("config.yaml")
TRIALS_CSV  = Path("output/optimization/lstm_trials.csv")

N_CANDIDATES    = 10   
N_FOLDS         = 4    
MIN_TRAIN_HOURS = 8760  

yaml_rt = YAML()
yaml_rt.preserve_quotes = True
yaml_rt.indent(mapping=2, sequence=4, offset=2)
yaml_rt.width = 4096

FIXED_FUTURE_COLS = ["hora_sin", "hora_cos"]

CANDIDATE_GROUPS = {
    "use_temperatura": ["temperatura"],
    "use_humedad":     ["humedad"],
    "use_temp_lags":   ["temp_lag_24h"],
    "use_hum_lags":    ["hum_lag_24h"],
    "use_feriado":     None,   
    "use_dia_anio":    ["dia_anio_sin", "dia_anio_cos"],
    "use_target_lags": ["target_lag_24"],
}

def load_config(config_path: Path = CONFIG_PATH) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml_rt.load(f)

def save_config(config: dict, config_path: Path = CONFIG_PATH) -> None:
    with open(config_path, "w", encoding="utf-8") as f:
        yaml_rt.dump(config, f)

def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    errors = actual - predicted

    mae  = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    nonzero = actual != 0
    mape = float(np.mean(np.abs(errors[nonzero] / actual[nonzero])) * 100) if nonzero.any() else float("nan")

    denom = np.abs(actual) + np.abs(predicted)
    nonzero_denom = denom != 0
    smape = float(np.mean(2 * np.abs(errors[nonzero_denom]) / denom[nonzero_denom]) * 100) if nonzero_denom.any() else float("nan")

    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    _INF_THRESHOLD = 1e12
    mae   = float("inf") if (np.isnan(mae)   or np.isinf(mae)   or mae   > _INF_THRESHOLD) else mae
    rmse  = float("inf") if (np.isnan(rmse)  or np.isinf(rmse)  or rmse  > _INF_THRESHOLD) else rmse
    mape  = float("inf") if (np.isnan(mape)  or np.isinf(mape)  or mape  > _INF_THRESHOLD) else mape
    smape = float("inf") if (np.isnan(smape) or np.isinf(smape) or smape > _INF_THRESHOLD) else smape

    return {"mae": mae, "rmse": rmse, "mape": mape, "smape": smape, "r2": r2}

def resolve_columns(df: pd.DataFrame, logger: logging.Logger) -> dict:
    temp_col = "temperatura"
    hum_col  = "humedad"

    if "es_feriado" in df.columns:
        feriado_col = "es_feriado"
    elif "es_laboral" in df.columns:
        feriado_col = "es_laboral"
        logger.warning(
            "Columna 'es_feriado' no encontrada. Usando 'es_laboral' como proxy. "
            "OJO: 'es_laboral' (día de semana vs fin de semana) NO es equivalente a "
            "'es_feriado' (fecha específica no laborable)."
        )
        print("  Usando 'es_laboral' como proxy de 'es_feriado' (ver log)")
    else:
        feriado_col = None
        logger.error("No se encontró columna de feriado ni de laboral en el dataset.")
        print("  No se encontró columna de feriado/laboral")

    return {"temp": temp_col, "hum": hum_col, "feriado": feriado_col}

def build_model(params_row, col_map, context_window, seed, device):
    """Construye un RNNModel LSTM a partir de una fila del CSV de trials.

    Devuelve (model, selected_cols).
    """
    hidden_size = int(params_row["hidden_size"])
    num_layers  = int(params_row["num_layers"])
    dropout     = float(params_row["dropout"])
    batch_size  = int(params_row["batch_size"])
    lr          = float(params_row["learning_rate"])
    epochs      = int(params_row["epochs"])

    selected_cols = list(FIXED_FUTURE_COLS)
    resolved = {
        "use_temperatura": [col_map["temp"]],
        "use_humedad":     [col_map["hum"]],
        "use_feriado":     [col_map["feriado"]] if col_map["feriado"] else [],
        "use_dia_anio":    ["dia_anio_sin", "dia_anio_cos"],
    }
    for flag, cols in resolved.items():
        flag_val = params_row.get(flag)
        if cols and pd.notna(flag_val) and bool(flag_val):
            selected_cols.extend(cols)

    model = RNNModel(
        model="LSTM",
        input_chunk_length=context_window,
        output_chunk_length=1,  
        training_length=context_window + 1,
        hidden_dim=hidden_size,
        n_rnn_layers=num_layers,
        dropout=dropout,
        batch_size=batch_size,
        optimizer_kwargs={"lr": lr},
        n_epochs=epochs,
        random_state=seed,
        pl_trainer_kwargs={
            "accelerator": "gpu" if device == "cuda" else "cpu",
            "devices": [0] if device == "cuda" else "auto",
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "logger": False,
        },
        force_reset=True,
        save_checkpoints=False,
    )
    return model, selected_cols

def make_factory(params_row, col_map, context_window, seed, device):
    """Devuelve una función que construye un modelo fresco (para cada fold)."""
    def factory():
        return build_model(params_row, col_map, context_window, seed, device)
    return factory

def build_cv_cuts(period_start, test_start, n_folds, min_train_hours):
    """Cortes de la CV temporal (expanding window) dentro de train+val.

    El primer corte deja al menos `min_train_hours` de entrenamiento.
    Los cortes se redondean a medianoche para que existan en el índice horario.
    """
    min_train = period_start + pd.Timedelta(hours=min_train_hours)
    span_days = max((test_start - min_train).days, n_folds)
    step_days = span_days // n_folds
    cuts = [min_train + pd.Timedelta(days=step_days * i) for i in range(n_folds)]
    cuts.append(test_start)
    return cuts

def run_cv(series_ts, future_covs_ts, model_factory, cuts, horizon, stride):
    """Expanding-window CV temporal dentro de train+val.

    Cada fold ajusta su propio Scaler con datos < cut (sin contaminación).
    Devuelve (fold_results, cv_rmse, pooled_metrics).
    """
    fold_results = []
    all_actual, all_pred = [], []

    for k in range(len(cuts) - 1):
        cut_k, cut_next = cuts[k], cuts[k + 1]
        model, selected_cols = model_factory()

        scaler_target = Scaler()
        scaler_target.fit(series_ts.drop_after(cut_k))
        series_scaled = scaler_target.transform(series_ts)

        scaler_future = Scaler()
        scaler_future.fit(future_covs_ts.drop_after(cut_k)[selected_cols])
        future_covs_scaled = scaler_future.transform(future_covs_ts[selected_cols])

        train_series = series_scaled.drop_after(cut_k)
        train_future = future_covs_scaled.drop_after(cut_k)

        model.fit(series=train_series, future_covariates=train_future, verbose=False)

        forecasts = model.historical_forecasts(
            series=series_scaled,
            future_covariates=future_covs_scaled,
            start=cut_k,
            forecast_horizon=horizon,
            stride=stride,
            retrain=False,
            overlap_end=False,
            last_points_only=False,
            verbose=False,
        )

        rmses = []
        for f in forecasts:
            if f.end_time() >= cut_next:
                continue
            actual_scaled = series_scaled.slice_intersect(f)
            actual_real = scaler_target.inverse_transform(actual_scaled).values().flatten()
            pred_real   = scaler_target.inverse_transform(f).values().flatten()
            rmses.append(float(np.sqrt(np.mean((actual_real - pred_real) ** 2))))
            all_actual.append(actual_real)
            all_pred.append(pred_real)

        fold_rmse = float(np.mean(rmses)) if rmses else float("nan")
        fold_results.append({
            "fold": k + 1,
            "train_until": str(cut_k),
            "eval_from": str(cut_k),
            "eval_to": str(cut_next),
            "n_windows": len(rmses),
            "rmse": fold_rmse,
        })
        del model
        gc.collect()

    valid = [r["rmse"] for r in fold_results if not np.isnan(r["rmse"])]
    cv_rmse = float(np.mean(valid)) if valid else float("inf")
    pooled = compute_metrics(np.concatenate(all_actual), np.concatenate(all_pred)) if all_actual else None
    return fold_results, cv_rmse, pooled

def evaluate_test(series_ts, future_covs_ts, model_factory, test_start, horizon, stride):
    """Refit con train+val completo y evalúa test UNA sola vez (holdout)."""
    model, selected_cols = model_factory()

    scaler_target = Scaler()
    scaler_target.fit(series_ts.drop_after(test_start))
    series_scaled = scaler_target.transform(series_ts)

    scaler_future = Scaler()
    scaler_future.fit(future_covs_ts.drop_after(test_start)[selected_cols])
    future_covs_scaled = scaler_future.transform(future_covs_ts[selected_cols])

    trainval_series = series_scaled.drop_after(test_start)
    trainval_future = future_covs_scaled.drop_after(test_start)

    model.fit(series=trainval_series, future_covariates=trainval_future, verbose=False)

    forecasts = model.historical_forecasts(
        series=series_scaled,
        future_covariates=future_covs_scaled,
        start=test_start,
        forecast_horizon=horizon,
        stride=stride,
        retrain=False,
        overlap_end=False,
        last_points_only=False,
        verbose=False,
    )

    all_actual, all_pred, all_dt = [], [], []
    for f in forecasts:
        actual_scaled = series_scaled.slice_intersect(f)
        actual_real = scaler_target.inverse_transform(actual_scaled).values().flatten()
        pred_real   = scaler_target.inverse_transform(f).values().flatten()

        k = min(len(actual_real), len(pred_real))
        all_actual.append(actual_real[:k])
        all_pred.append(pred_real[:k])

        all_dt.append(pd.DatetimeIndex(actual_scaled.time_index)[:k])

    if not all_actual:
        return None, 0, np.array([]), np.array([]), pd.DatetimeIndex([])
    actual_cat = np.concatenate(all_actual)
    pred_cat   = np.concatenate(all_pred)
    metrics = compute_metrics(actual_cat, pred_cat)
    return metrics, len(all_actual), actual_cat, pred_cat, all_dt[0].append(all_dt[1:])

def _round_metric(v):
    """Redondea a 4 decimales; None/NaN se guardan como None."""
    if v is None:
        return None
    v = float(v)
    return float(round(v, 4)) if not np.isnan(v) else None

def main():
    config = load_config(CONFIG_PATH)
    seed   = config.get("project", {}).get("random_seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")

    LOG_DIR   = Path(config.get("output", {}).get("logs", "output/logs"))
    EVAL_DIR  = Path(config.get("output", {}).get("evaluation", "output/evaluation"))
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("lstm_evaluation")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_DIR / "lstm_evaluation.log")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    dataset_cfg = config.get("dataset", {})
    csv_path    = dataset_cfg.get("input_file_lstm")
    date_col    = dataset_cfg.get("datetime", {}).get("date_column", "FECHA")
    hour_col    = dataset_cfg.get("datetime", {}).get("hour_column", "Horario")
    target_col  = dataset_cfg.get("target", "TOTAL_SIP")
    period_cfg  = dataset_cfg.get("period", {})

    print(f"Cargando datos desde {csv_path}...")
    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(df[date_col]) + pd.to_timedelta(df[hour_col], unit="h")
    df = df.set_index("datetime").sort_index()

    if period_cfg.get("start") and period_cfg.get("end"):
        start_ts = pd.to_datetime(str(period_cfg["start"]))
        end_ts   = pd.to_datetime(str(period_cfg["end"]))
        df = df.loc[start_ts:end_ts]

    col_map = resolve_columns(df, logger)

    required_cols = (
        FIXED_FUTURE_COLS
        + [col_map["temp"], col_map["hum"]]
        + ["dia_anio_sin", "dia_anio_cos"]
        + ["temp_lag_24h", "hum_lag_24h", "target_lag_24"]
    )
    if col_map["feriado"]:
        required_cols.append(col_map["feriado"])
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Error: columnas faltantes: {missing}")
        sys.exit(1)

    period_start  = pd.to_datetime(str(period_cfg.get("start", "2023-01-01")))
    period_end    = pd.to_datetime(str(period_cfg.get("end", "2026-01-01")))
    split_cfg     = config.get("split", {})
    train_ratio   = split_cfg.get("train_ratio", 0.70)
    val_ratio     = split_cfg.get("validation_ratio", 0.15)

    period_hours  = (period_end - period_start).total_seconds() / 3600
    train_hours   = int(period_hours * train_ratio)
    val_hours     = int(period_hours * val_ratio)

    VAL_START  = period_start + pd.Timedelta(hours=train_hours)
    TEST_START = period_start + pd.Timedelta(hours=train_hours + val_hours)
    VAL_END    = TEST_START

    n_total        = len(df)
    val_start_idx  = int(df.index.searchsorted(VAL_START))
    test_start_idx = int(df.index.searchsorted(TEST_START))

    bt_cfg           = config.get("backtesting", {})
    FORECAST_HORIZON = bt_cfg.get("forecast_horizon", 24)
    STRIDE           = bt_cfg.get("stride", 24)
    CONTEXT_WINDOW   = bt_cfg.get("context_window", 24 * 7)

    n_train = val_start_idx
    n_val   = test_start_idx - val_start_idx
    n_test  = n_total - test_start_idx

    print(f"Train: {n_train} obs (hasta {VAL_START})")
    print(f"Val:   {n_val} obs ({VAL_START} -> {VAL_END})")
    print(f"Test:  {n_test} obs (desde {TEST_START}, reservado)")
    print(f"Walk-forward: horizon={FORECAST_HORIZON}h, stride={STRIDE}h, context_window={CONTEXT_WINDOW}h")

    all_future_cols = (
        FIXED_FUTURE_COLS
        + [col_map["temp"], col_map["hum"], "dia_anio_sin", "dia_anio_cos"]
        + ["temp_lag_24h", "hum_lag_24h", "target_lag_24"]
        + ([col_map["feriado"]] if col_map["feriado"] else [])
    )
    seen = set()
    all_future_cols = [c for c in all_future_cols if not (c in seen or seen.add(c))]

    series_ts      = TimeSeries.from_dataframe(df[[target_col]], freq="h")
    future_covs_ts = TimeSeries.from_dataframe(df[all_future_cols], freq="h")

    period_start = df.index[0]
    cuts = build_cv_cuts(period_start, TEST_START, N_FOLDS, MIN_TRAIN_HOURS)
    print(f"\nCV temporal ({N_FOLDS} folds, expanding window, dentro de train+val):")
    for k in range(len(cuts) - 1):
        print(f"  fold {k + 1}: fit < {cuts[k]}  |  eval [{cuts[k]}, {cuts[k + 1]})")

    if not TRIALS_CSV.exists():
        print(f"\nError: no se encontró {TRIALS_CSV}. Corré primero src/optimization/lstm_optuna.py")
        sys.exit(1)

    trials_df = pd.read_csv(TRIALS_CSV)
    trials_df["val_rmse"] = pd.to_numeric(trials_df["val_rmse"], errors="coerce")

    valid = trials_df.dropna(subset=["val_rmse"])
    valid = valid[valid["val_rmse"] != float("inf")]

    if len(valid) < N_CANDIDATES:
        print(f"\nAdvertencia: solo hay {len(valid)} trials válidos (se pidieron {N_CANDIDATES}).")
        N_CAND = len(valid)
    else:
        N_CAND = N_CANDIDATES

    candidates = valid.nsmallest(N_CAND, "val_rmse")

    print(f"\n=== CANDIDATOS (top-{N_CAND} por val_rmse) ===")
    for _, r in candidates.iterrows():
        print(f"  trial #{int(r['trial_number']):>3d}  val_rmse={float(r['val_rmse']):8.4f}  "
              f"test_rmse(orientativo)={float(r['test_rmse']):8.4f}")

    print(f"\nCorriendo CV temporal para {N_CAND} candidatos x {N_FOLDS} folds...")
    print("=" * 80)

    results = []
    for _, row in candidates.iterrows():
        trial_num = int(row["trial_number"])
        factory = make_factory(row, col_map, CONTEXT_WINDOW, seed, device)
        t0 = time.time()
        fold_results, cv_rmse, pooled = run_cv(
            series_ts, future_covs_ts, factory, cuts,
            FORECAST_HORIZON, STRIDE,
        )
        elapsed = time.time() - t0

        fold_txt = " | ".join(
            f"f{r['fold']}={r['rmse']:.2f}({r['n_windows']}w)" for r in fold_results
        )
        print(f"[trial {trial_num:>3d}] CV_RMSE={cv_rmse:8.4f}  ({elapsed:5.0f}s)  {fold_txt}")

        selected_cols = list(FIXED_FUTURE_COLS)
        resolved = {
            "use_temperatura": [col_map["temp"]],
            "use_humedad":     [col_map["hum"]],
            "use_feriado":     [col_map["feriado"]] if col_map["feriado"] else [],
            "use_dia_anio":    ["dia_anio_sin", "dia_anio_cos"],
        }
        for flag, cols in resolved.items():
            flag_val = row.get(flag)
            if cols and pd.notna(flag_val) and bool(flag_val):
                selected_cols.extend(cols)
        candidate_cols = [c for c in selected_cols if c not in FIXED_FUTURE_COLS]

        results.append({
            "trial_number": trial_num,
            "val_rmse": float(row["val_rmse"]),
            "cv_rmse": cv_rmse,
            "cv_rmse_pooled": pooled["rmse"] if pooled else None,
            "cv_mae":  pooled["mae"]  if pooled else None,
            "cv_mape": pooled["mape"] if pooled else None,
            "cv_smape": pooled["smape"] if pooled else None,
            "cv_r2":   pooled["r2"]   if pooled else None,
            "hidden_size": int(row["hidden_size"]),
            "num_layers": int(row["num_layers"]),
            "dropout": float(row["dropout"]),
            "batch_size": int(row["batch_size"]),
            "learning_rate": float(row["learning_rate"]),
            "epochs": int(row["epochs"]),
            "selected_features": candidate_cols,
            "cv_seconds": round(elapsed, 1),
        })

    results_df = pd.DataFrame(results).sort_values("cv_rmse").reset_index(drop=True)
    horizon = config.get("backtesting", {}).get("forecast_horizon", 24)
    suffix = f"_{horizon}h"
    cv_csv_name = f"lstm_cv_top10{suffix}.csv"
    out_csv = EVAL_DIR / cv_csv_name
    results_df.to_csv(out_csv, index=False)
    print(f"\nTabla de CV guardada en: {out_csv}")
    logger.info(f"CV top-{N_CAND} guardado en {out_csv}")

    print("\n" + "=" * 80)
    print("RANKING POR CV_RMSE (criterio de selección, test NO se usa)")
    print("=" * 80)
    for i, r in results_df.iterrows():
        print(f"  [{i + 1:>2d}] trial #{int(r['trial_number']):>3d}  "
              f"cv_rmse={float(r['cv_rmse']):8.4f}  val_rmse={float(r['val_rmse']):8.4f}")

    winner = results_df.iloc[0]
    winner_trial = int(winner["trial_number"])
    winner_row = candidates[candidates["trial_number"] == winner_trial].iloc[0]
    factory = make_factory(winner_row, col_map, CONTEXT_WINDOW, seed, device)

    print("\n" + "=" * 80)
    print(f"EVALUACIÓN FINAL SOBRE TEST (holdout, UNA sola vez)")
    print(f"Ganador por CV: trial #{winner_trial}")
    print("=" * 80)

    test_metrics, n_windows_test, test_actual, test_pred, test_datetime = evaluate_test(
        series_ts, future_covs_ts, factory, TEST_START,
        FORECAST_HORIZON, STRIDE,
    )

    if test_metrics:
        print(f"\n  Métricas en test ({n_windows_test} ventanas):")
        for k, v in test_metrics.items():
            print(f"    {k:6s} = {v:.4f}")
        logger.info(f"Test metrics trial #{winner_trial}: {test_metrics}")

        horizon = config.get("backtesting", {}).get("forecast_horizon", 24)
        pred_df = pd.DataFrame({
            "datetime": test_datetime,
            "actual": test_actual,
            "predicted": test_pred,
        })
        pred_csv = EVAL_DIR / f"lstm_test_predictions_{horizon}h.csv"
        pred_df.to_csv(pred_csv, index=False)

        write_forecast_meta(
            pred_csv, model="lstm", horizon=FORECAST_HORIZON, stride=STRIDE,
            context_window=CONTEXT_WINDOW, test_start=TEST_START,
            n_windows=int(n_windows_test), n_predictions=len(pred_df),
        )
        print(f"  Predicciones del test guardadas en: {pred_csv}")
    else:
        print("\n  No se generaron ventanas de test.")

    winner_info = {
        "trial_number": winner_trial,
        "cv_rmse": float(winner["cv_rmse"]),
        "val_rmse": float(winner["val_rmse"]),
        "hyperparameters": {
            "hidden_size": int(winner["hidden_size"]),
            "num_layers": int(winner["num_layers"]),
            "dropout": float(winner["dropout"]),
            "batch_size": int(winner["batch_size"]),
            "learning_rate": float(winner["learning_rate"]),
            "epochs": int(winner["epochs"]),
        },
        "fixed_features": FIXED_FUTURE_COLS,
        "selected_features": winner["selected_features"],
        "test_metrics": test_metrics,
        "n_windows_test": n_windows_test,
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_yaml = EVAL_DIR / f"lstm_cv_winner_{horizon}h.yaml"
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml_rt.dump(winner_info, f)
    print(f"\n  Detalle del ganador guardado en: {out_yaml}")

    horizon = config.get("backtesting", {}).get("forecast_horizon", 24)
    horizon_key = f"best_{horizon}h"
    if "lstm" not in config:
        config["lstm"] = {}

    new_best = {}
    new_best["hidden_size"]         = int(winner["hidden_size"])
    new_best["num_layers"]          = int(winner["num_layers"])
    new_best["dropout"]             = float(round(float(winner["dropout"]), 4))
    new_best["batch_size"]          = int(winner["batch_size"])
    new_best["learning_rate"]       = float(winner["learning_rate"])
    new_best["epochs"]              = int(winner["epochs"])
    new_best["input_chunk_length"]  = int(CONTEXT_WINDOW)

    new_best["output_chunk_length"] = 1
    new_best["forecast_horizon"]    = int(FORECAST_HORIZON)
    new_best["forecast_strategy"]   = "autoregressive"
    new_best["fixed_features"]      = FIXED_FUTURE_COLS
    new_best["selected_features"]   = list(winner["selected_features"])
    new_best["rmse"]                = _round_metric(winner["val_rmse"])

    new_best["cv_rmse"]        = _round_metric(winner["cv_rmse"])
    new_best["cv_rmse_pooled"] = _round_metric(winner["cv_rmse_pooled"])
    new_best["cv_mae"]         = _round_metric(winner["cv_mae"])
    new_best["cv_mape"]        = _round_metric(winner["cv_mape"])
    new_best["cv_smape"]       = _round_metric(winner["cv_smape"])
    new_best["cv_r2"]          = _round_metric(winner["cv_r2"])

    new_best["rmse_test"]  = _round_metric(test_metrics["rmse"]  if test_metrics else None)
    new_best["mae_test"]   = _round_metric(test_metrics["mae"]   if test_metrics else None)
    new_best["mape_test"]  = _round_metric(test_metrics["mape"]  if test_metrics else None)
    new_best["smape_test"] = _round_metric(test_metrics["smape"] if test_metrics else None)
    new_best["r2_test"]    = _round_metric(test_metrics["r2"]    if test_metrics else None)
    new_best["trial_number"] = int(winner_trial)
    new_best["datetime"]     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    config["lstm"][horizon_key] = new_best
    save_config(config, CONFIG_PATH)
    print(f"\n  Ganador escrito en 'lstm.{horizon_key}' en {CONFIG_PATH}")
    logger.info(f"Ganador trial #{winner_trial} escrito en lstm.{horizon_key}")

    print("\nNota: el test_rmse reportado es el error real del modelo elegido por CV.")

if __name__ == "__main__":
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.getLogger("darts").setLevel(logging.ERROR)
    main()