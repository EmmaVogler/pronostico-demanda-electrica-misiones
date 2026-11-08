import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
import gc
import time
import logging
import warnings
import ast
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
from darts.models import Chronos2Model

warnings.filterwarnings("ignore")

CONFIG_PATH = Path("config.yaml")
TRIALS_CSV  = Path("output/optimization/chronos2_trials.csv")

N_CANDIDATES    = 10   
N_FOLDS         = 4    
MIN_TRAIN_HOURS = 8760  

yaml_rt = YAML()
yaml_rt.preserve_quotes = True
yaml_rt.indent(mapping=2, sequence=4, offset=2)
yaml_rt.width = 4096

CANDIDATE_PAST_GROUPS = {}

CANDIDATE_FUTURE_GROUPS = {
    "use_dia_semana": ["dia_semana"],
    "use_es_laboral": ["es_laboral"],
    "use_temp_f24":   ["TEMP_F_24"],
    "use_hum_f24":    ["HUM_F_24"],
    "use_temp_lags":  ["temp_lag_24h"],
    "use_hum_lags":   ["hum_lag_24h"],
    "use_target_lags": ["target_lag_24"],
}

ALL_PAST_COLS = list(dict.fromkeys(
    c for cols in CANDIDATE_PAST_GROUPS.values() for c in cols
))
ALL_FUTURE_COLS = list(dict.fromkeys(
    c for cols in CANDIDATE_FUTURE_GROUPS.values() for c in cols
))

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

def _round_metric(v):
    """Redondea a 4 decimales; None/NaN se guardan como None."""
    if v is None:
        return None
    v = float(v)
    return float(round(v, 4)) if not np.isnan(v) else None

def _parse_list(s):
    """Parsea una celda del CSV que contiene una lista Python como string."""
    if isinstance(s, str) and s.strip():
        try:
            return ast.literal_eval(s)
        except Exception:
            return []
    return []

def build_model(row, context_window, forecast_horizon, device):
    """Construye un Chronos2Model a partir de una fila del CSV de trials.

    Devuelve (model, past_cols, future_cols).
    """
    hub_model_name = str(row["hub_model_name"])
    past_cols   = _parse_list(row.get("selected_past_cols"))
    future_cols = _parse_list(row.get("selected_future_cols"))

    model = Chronos2Model(
        input_chunk_length=context_window,
        output_chunk_length=forecast_horizon,
        hub_model_name=hub_model_name,
        pl_trainer_kwargs={
            "accelerator": "gpu" if device == "cuda" else "cpu",
            "devices": [0] if device == "cuda" else "auto",
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "logger": False,
        },
    )
    return model, past_cols, future_cols

def make_factory(row, context_window, forecast_horizon, device):
    """Devuelve una función que construye un modelo fresco (para cada fold)."""
    def factory():
        return build_model(row, context_window, forecast_horizon, device)
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

def run_cv(series, past_covs, future_covs, model_factory, cuts, horizon, stride, device):
    """Expanding-window CV temporal dentro de train+val.

    fold k: fit con datos < cuts[k], walk-forward predict en [cuts[k], cuts[k+1]).
    Devuelve (fold_results, cv_rmse, pooled_metrics).
    """
    fold_results = []
    all_actual, all_pred = [], []

    for k in range(len(cuts) - 1):
        cut_k, cut_next = cuts[k], cuts[k + 1]
        model, past_cols, future_cols = model_factory()

        train_series = series.drop_after(cut_k)
        train_past   = past_covs.drop_after(cut_k)[past_cols] if past_covs is not None and past_cols else None
        train_future = future_covs.drop_after(cut_k)[future_cols] if future_covs is not None and future_cols else None

        model.fit(
            series=train_series,
            past_covariates=train_past,
            future_covariates=train_future,
            verbose=False,
        )

        hf_kwargs = {
            "series": series,
            "start": cut_k,
            "forecast_horizon": horizon,
            "stride": stride,
            "retrain": False,
            "overlap_end": False,
            "last_points_only": False,
            "verbose": False,
            "show_warnings": False,
        }
        if past_cols:
            hf_kwargs["past_covariates"] = past_covs[past_cols]
        if future_cols:
            hf_kwargs["future_covariates"] = future_covs[future_cols]

        forecasts = model.historical_forecasts(**hf_kwargs)

        rmses = []
        for f in forecasts:
            if f.end_time() >= cut_next:
                continue
            actual = series.slice_intersect(f)
            act = actual.values().flatten()
            pred = f.values().flatten()
            rmses.append(float(np.sqrt(np.mean((act - pred) ** 2))))
            all_actual.append(act)
            all_pred.append(pred)

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
        if device == "cuda":
            torch.cuda.empty_cache()

    valid = [r["rmse"] for r in fold_results if not np.isnan(r["rmse"])]
    cv_rmse = float(np.mean(valid)) if valid else float("inf")
    pooled = compute_metrics(np.concatenate(all_actual), np.concatenate(all_pred)) if all_actual else None
    return fold_results, cv_rmse, pooled

def evaluate_test(series, past_covs, future_covs, model_factory, test_start, horizon, stride):
    """Refit con train+val completo y evalúa test UNA sola vez (holdout)."""
    model, past_cols, future_cols = model_factory()

    trainval_series = series.drop_after(test_start)
    trainval_past   = past_covs.drop_after(test_start)[past_cols] if past_covs is not None and past_cols else None
    trainval_future = future_covs.drop_after(test_start)[future_cols] if future_covs is not None and future_cols else None

    model.fit(
        series=trainval_series,
        past_covariates=trainval_past,
        future_covariates=trainval_future,
        verbose=False,
    )

    hf_kwargs = {
        "series": series,
        "start": test_start,
        "forecast_horizon": horizon,
        "stride": stride,
        "retrain": False,
        "overlap_end": False,
        "last_points_only": False,
        "verbose": False,
        "show_warnings": False,
    }
    if past_cols:
        hf_kwargs["past_covariates"] = past_covs[past_cols]
    if future_cols:
        hf_kwargs["future_covariates"] = future_covs[future_cols]

    forecasts = model.historical_forecasts(**hf_kwargs)

    all_actual, all_pred, all_dt = [], [], []
    for f in forecasts:
        actual = series.slice_intersect(f)
        a_vals = actual.values().flatten()
        p_vals = f.values().flatten()

        k = min(len(a_vals), len(p_vals))
        all_actual.append(a_vals[:k])
        all_pred.append(p_vals[:k])

        all_dt.append(pd.DatetimeIndex(actual.time_index)[:k])

    if not all_actual:
        return None, 0, np.array([]), np.array([]), pd.DatetimeIndex([])
    actual_cat = np.concatenate(all_actual)
    pred_cat   = np.concatenate(all_pred)
    metrics = compute_metrics(actual_cat, pred_cat)
    return metrics, len(all_actual), actual_cat, pred_cat, all_dt[0].append(all_dt[1:])

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

    logger = logging.getLogger("chronos2_evaluation")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_DIR / "chronos2_evaluation.log")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    dataset_cfg = config.get("dataset", {})
    csv_path    = dataset_cfg.get("input_file_chronos2")
    target_col  = dataset_cfg.get("target", "TOTAL_SIP")
    period_cfg  = dataset_cfg.get("period", {})

    print(f"Cargando datos desde {csv_path}...")
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()

    period_start = pd.to_datetime(str(period_cfg.get("start", "2023-01-01")))
    period_end   = pd.to_datetime(str(period_cfg.get("end", "2026-01-01")))

    split_cfg   = config.get("split", {})
    train_ratio = split_cfg.get("train_ratio", 0.70)
    val_ratio   = split_cfg.get("validation_ratio", 0.15)

    df = df.loc[(df.index >= period_start) & (df.index <= period_end)]

    period_hours = (period_end - period_start).total_seconds() / 3600
    train_hours  = int(period_hours * train_ratio)
    val_hours    = int(period_hours * val_ratio)

    VAL_START  = period_start + pd.Timedelta(hours=train_hours)
    TEST_START = period_start + pd.Timedelta(hours=train_hours + val_hours)
    VAL_END    = TEST_START

    n_total        = len(df)
    val_start_idx  = int(df.index.searchsorted(VAL_START))
    test_start_idx = int(df.index.searchsorted(TEST_START))

    missing = [c for c in [target_col] + ALL_PAST_COLS + ALL_FUTURE_COLS if c not in df.columns]
    if missing:
        print(f"Advertencia: columnas faltantes: {missing}")

    bt_cfg           = config.get("backtesting", {})
    FORECAST_HORIZON = bt_cfg.get("forecast_horizon", 24)
    STRIDE           = bt_cfg.get("stride", 24)
    CONTEXT_WINDOW   = bt_cfg.get("context_window", 168)

    n_train = len(df[df.index < VAL_START])
    n_val   = len(df[(df.index >= VAL_START) & (df.index < VAL_END)])
    n_test  = len(df[df.index >= TEST_START])

    print(f"Train: {n_train} obs (hasta {VAL_START})")
    print(f"Val:   {n_val} obs ({VAL_START} -> {VAL_END})")
    print(f"Test:  {n_test} obs (desde {TEST_START}, reservado)")
    print(f"Walk-forward: horizon={FORECAST_HORIZON}h, stride={STRIDE}h, context_window={CONTEXT_WINDOW}h")

    series_ts      = TimeSeries.from_dataframe(df[[target_col]], freq="h")
    past_covs_ts   = TimeSeries.from_dataframe(df[ALL_PAST_COLS], freq="h") if ALL_PAST_COLS else None
    future_covs_ts = TimeSeries.from_dataframe(df[ALL_FUTURE_COLS], freq="h") if ALL_FUTURE_COLS else None

    cuts = build_cv_cuts(period_start, TEST_START, N_FOLDS, MIN_TRAIN_HOURS)
    print(f"\nCV temporal ({N_FOLDS} folds, expanding window, dentro de train+val):")
    for k in range(len(cuts) - 1):
        print(f"  fold {k + 1}: fit < {cuts[k]}  |  eval [{cuts[k]}, {cuts[k + 1]})")

    if not TRIALS_CSV.exists():
        print(f"\nError: no se encontró {TRIALS_CSV}. Corré primero src/optimization/chronos2_optuna.py")
        sys.exit(1)

    trials_df = pd.read_csv(TRIALS_CSV)
    trials_df["rmse_val_num"] = pd.to_numeric(trials_df["rmse_val"], errors="coerce")

    valid = trials_df.dropna(subset=["rmse_val_num"])
    valid = valid[valid["rmse_val_num"] != float("inf")]

    def _config_key(row):
        pas = tuple(sorted(_parse_list(row.get("selected_past_cols"))))
        fut = tuple(sorted(_parse_list(row.get("selected_future_cols"))))
        return (pas, fut)

    valid = valid.assign(_config=valid.apply(_config_key, axis=1))

    por_config = (valid.sort_values(["rmse_val_num", "trial_number"])
                       .drop_duplicates("_config", keep="first"))
    print(f"\nConfiguraciones distintas: {len(por_config)} entre {len(valid)} trials con RMSE finito "
          f"(se descartan {len(valid) - len(por_config)} trials que replican otra configuración).")

    if len(por_config) < N_CANDIDATES:
        print(f"\nAdvertencia: solo hay {len(por_config)} configuraciones distintas (se pidieron {N_CANDIDATES}).")
        N_CAND = len(por_config)
    else:
        N_CAND = N_CANDIDATES

    candidates = por_config.nsmallest(N_CAND, "rmse_val_num").drop(columns="_config")

    print(f"\n=== CANDIDATOS (top-{N_CAND} configuraciones por rmse_val) ===")
    for _, r in candidates.iterrows():
        test_rmse = r.get("rmse_test")
        test_rmse_s = "?" if pd.isna(test_rmse) else f"{float(test_rmse):8.4f}"
        covars = ", ".join(_parse_list(r.get("selected_future_cols"))) or "(ninguna)"
        print(f"  trial #{int(r['trial_number']):>3d}  rmse_val={float(r['rmse_val_num']):8.4f}  "
              f"covariables=[{covars}]  test_rmse(orientativo)={test_rmse_s}")

    print(f"\nCorriendo CV temporal para {N_CAND} candidatos x {N_FOLDS} folds...")
    print("=" * 80)

    results = []
    for _, row in candidates.iterrows():
        trial_num = int(row["trial_number"])
        factory = make_factory(row, CONTEXT_WINDOW, FORECAST_HORIZON, device)
        t0 = time.time()
        fold_results, cv_rmse, pooled = run_cv(
            series_ts, past_covs_ts, future_covs_ts, factory, cuts,
            FORECAST_HORIZON, STRIDE, device,
        )
        elapsed = time.time() - t0

        fold_txt = " | ".join(
            f"f{r['fold']}={r['rmse']:.2f}({r['n_windows']}w)" for r in fold_results
        )
        print(f"[trial {trial_num:>3d}] CV_RMSE={cv_rmse:8.4f}  ({elapsed:5.0f}s)  {fold_txt}")

        past_cols   = _parse_list(row.get("selected_past_cols"))
        future_cols = _parse_list(row.get("selected_future_cols"))

        results.append({
            "trial_number": trial_num,
            "val_rmse": float(row["rmse_val_num"]),
            "cv_rmse": cv_rmse,
            "cv_rmse_pooled": pooled["rmse"] if pooled else None,
            "cv_mae":  pooled["mae"]  if pooled else None,
            "cv_mape": pooled["mape"] if pooled else None,
            "cv_smape": pooled["smape"] if pooled else None,
            "cv_r2":   pooled["r2"]   if pooled else None,
            "hub_model_name": str(row["hub_model_name"]),
            "past_covariates": past_cols,
            "future_covariates": future_cols,
            "cv_seconds": round(elapsed, 1),
        })

    results_df = pd.DataFrame(results).sort_values("cv_rmse").reset_index(drop=True)
    horizon = config.get("backtesting", {}).get("forecast_horizon", 24)
    suffix = f"_{horizon}h"
    cv_csv_name = f"chronos2_cv_top10{suffix}.csv"
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
    factory = make_factory(winner_row, CONTEXT_WINDOW, FORECAST_HORIZON, device)

    print("\n" + "=" * 80)
    print(f"EVALUACIÓN FINAL SOBRE TEST (holdout, UNA sola vez)")
    print(f"Ganador por CV: trial #{winner_trial}")
    print("=" * 80)

    test_metrics, n_windows_test, test_actual, test_pred, test_datetime = evaluate_test(
        series_ts, past_covs_ts, future_covs_ts, factory, TEST_START,
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
        pred_csv = EVAL_DIR / f"chronos2_test_predictions_{horizon}h.csv"
        pred_df.to_csv(pred_csv, index=False)

        write_forecast_meta(
            pred_csv, model="chronos2", horizon=FORECAST_HORIZON, stride=STRIDE,
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
            "hub_model_name": str(winner["hub_model_name"]),
            "input_chunk_length": int(CONTEXT_WINDOW),
            "output_chunk_length": int(FORECAST_HORIZON),
        },
        "past_covariates": list(winner["past_covariates"]),
        "future_covariates": list(winner["future_covariates"]),
        "test_metrics": test_metrics,
        "n_windows_test": n_windows_test,
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_yaml = EVAL_DIR / f"chronos2_cv_winner_{horizon}h.yaml"
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml_rt.dump(winner_info, f)
    print(f"\n  Detalle del ganador guardado en: {out_yaml}")

    horizon = config.get("backtesting", {}).get("forecast_horizon", 24)
    horizon_key = f"best_{horizon}h"
    if "chronos2" not in config:
        config["chronos2"] = {}

    new_best = {}
    new_best["hub_model_name"]      = str(winner["hub_model_name"])
    new_best["past_covariates"]     = list(winner["past_covariates"])
    new_best["future_covariates"]   = list(winner["future_covariates"])
    new_best["input_chunk_length"]  = int(CONTEXT_WINDOW)
    new_best["output_chunk_length"] = int(FORECAST_HORIZON)
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

    config["chronos2"][horizon_key] = new_best
    save_config(config, CONFIG_PATH)
    print(f"\n  Ganador escrito en 'chronos2.{horizon_key}' en {CONFIG_PATH}")
    logger.info(f"Ganador trial #{winner_trial} escrito en chronos2.{horizon_key}")

    print("\nNota: el test_rmse reportado es el error real del modelo elegido por CV.")

if __name__ == "__main__":
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.getLogger("darts").setLevel(logging.ERROR)
    main()