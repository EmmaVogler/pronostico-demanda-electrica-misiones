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
import ast
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from ruamel.yaml import YAML

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast_meta import write_forecast_meta  
from statsforecast.models import ARIMA
import statsforecast.arima as _sf_arima

warnings.filterwarnings("ignore", category=UserWarning)

CONFIG_PATH = Path("config.yaml")
TRIALS_CSV  = Path("output/optimization/sarimax_trials.csv")

N_CANDIDATES    = 10   
N_FOLDS         = 4    
MIN_TRAIN_HOURS = 8760  

yaml_rt = YAML()
yaml_rt.preserve_quotes = True
yaml_rt.indent(mapping=2, sequence=4, offset=2)
yaml_rt.width = 4096

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

def _round_metric(value) -> float:
    if value is None:
        return None
    value = float(value)
    if np.isnan(value) or np.isinf(value):
        return None
    return float(round(value, 4))

def row_to_float(v):
    """Convierte un valor de celda del CSV a float; None/NaN/inf -> None."""
    if v is None:
        return None
    try:
        v = float(v)
    except Exception:
        return None
    if np.isnan(v) or np.isinf(v):
        return None
    return v

def _build_fourier_df(
    index: pd.DatetimeIndex,
    fourier_periods: dict,
    k_daily: int,
    k_weekly: int,
    k_yearly: int,
) -> pd.DataFrame:
    """Construye los términos de Fourier (sin/cos) sobre un índice horario continuo."""
    t_hours = (index - index[0]).total_seconds() / 3600.0
    k_map = {"daily": k_daily, "weekly": k_weekly, "yearly": k_yearly}
    cols = {}
    for period_name, period in fourier_periods.items():
        for k in range(1, k_map[period_name] + 1):
            cols[f"fourier_{period_name}_sin_{k}"] = np.sin(2 * np.pi * k * t_hours / period)
            cols[f"fourier_{period_name}_cos_{k}"] = np.cos(2 * np.pi * k * t_hours / period)
    return pd.DataFrame(cols, index=index)

def _fixed_window_forecast(
    y_all: np.ndarray,
    X_all: np.ndarray,
    abs_pos: int,
    context_window: int,
    forecast_horizon: int,
    order: tuple,
    seasonal_order: tuple,
    season_length: int,
    fixed: dict,
    method: str = "CSS",
) -> np.ndarray:
    """Re-siembra el estado del ARIMA con parámetros fijos sobre la ventana de
    contexto y genera el pronóstico del siguiente horizonte (equivalente a
    statsmodels ``filter(fitted_params)`` + ``get_forecast``)."""
    window_start = max(0, abs_pos - context_window)
    ctx_y = y_all[window_start:abs_pos]
    ctx_X = X_all[window_start:abs_pos]
    fc_X = X_all[abs_pos : abs_pos + forecast_horizon]

    m = ARIMA(
        order=order,
        season_length=season_length,
        seasonal_order=seasonal_order,
        fixed=dict(fixed),
        method=method,
    )
    m.fit(ctx_y, ctx_X)
    fc = _sf_arima.forecast_arima(m.model_, h=forecast_horizon, xreg=fc_X)
    return fc["mean"]

def build_cv_cuts_idx(test_start_idx: int, n_folds: int, min_train_hours: int) -> list:
    """Cortes de la CV temporal (expanding window) en índices enteros.

    Frecuencia horaria -> índice == horas. El primer corte deja al menos
    `min_train_hours` de entrenamiento.
    """
    min_train_idx = min_train_hours
    span = max(test_start_idx - min_train_idx, n_folds)
    step = span // n_folds
    cuts = [min_train_idx + step * i for i in range(n_folds)]
    cuts.append(test_start_idx)
    return cuts

def _param_default(search_cfg: dict, name: str, fallback: int) -> int:
    """Valor fijo de un parámetro si su cota de búsqueda tiene un solo elemento.

    Optuna solo guarda en el CSV los parámetros que muestrea; si la cota es
    [x] (fija), la columna params_<name> no existe y hay que usar x.
    """
    bounds = search_cfg.get(name, [fallback])
    if isinstance(bounds, (list, tuple)) and len(bounds) == 1:
        return int(bounds[0])
    return fallback

def build_model_params(row, fixed_features: list, candidates_cfg: dict, search_cfg: dict = None) -> dict:
    """Reconstruye los parámetros del modelo desde una fila del CSV de trials
    (formato Optuna estándar: params_*, user_attrs_*).

    Los parámetros con cota fija (p.ej. d: [0]) no aparecen como columnas
    params_* en el CSV; se usa el valor fijo de la config como default.
    """
    search_cfg = search_cfg or {}
    p = int(row.get("params_p", _param_default(search_cfg, "p", 0)))
    d = int(row.get("params_d", _param_default(search_cfg, "d", 0)))
    q = int(row.get("params_q", _param_default(search_cfg, "q", 0)))
    P = int(row.get("params_P", _param_default(search_cfg, "P", 0)))
    D = int(row.get("params_D", _param_default(search_cfg, "D", 0)))
    Q = int(row.get("params_Q", _param_default(search_cfg, "Q", 0)))
    s = int(row.get("params_seasonal_period", _param_default(search_cfg, "seasonal_period", 24)))
    k_daily  = int(row.get("params_k_daily", _param_default(search_cfg, "k_daily", 1)))
    k_weekly = int(row.get("params_k_weekly", _param_default(search_cfg, "k_weekly", 1)))
    k_yearly = int(row.get("params_k_yearly", _param_default(search_cfg, "k_yearly", 1)))

    selected_features = list(fixed_features)
    for group_name, group_info in candidates_cfg.items():
        flag_name = group_info.get("flag", f"use_{group_name}")
        feats = group_info.get("features", [])
        flag_val = row.get(f"params_{flag_name}")
        if pd.notna(flag_val) and bool(flag_val):
            selected_features.extend(feats)

    return {
        "order": (p, d, q),
        "seasonal_order": (P, D, Q),
        "season_length": s,
        "k_daily": k_daily,
        "k_weekly": k_weekly,
        "k_yearly": k_yearly,
        "features": selected_features,
    }

def run_cv(target, covariates, fourier_periods, model_params, cuts, horizon, stride, context_window, method):
    """Expanding-window CV temporal dentro de train+val (índices enteros).

    fold k: fit ARIMA con datos < cuts[k], walk-forward predict en [cuts[k], cuts[k+1]).
    Devuelve (fold_results, cv_rmse, pooled_metrics).
    """
    fold_results = []
    all_actual, all_pred = [], []

    y_all = target.to_numpy(dtype=float)
    fourier_df = _build_fourier_df(
        target.index, fourier_periods,
        model_params["k_daily"], model_params["k_weekly"], model_params["k_yearly"],
    )
    cov_features = [f for f in model_params["features"] if f in covariates.columns]
    X_all = np.hstack(
        [covariates[cov_features].to_numpy(dtype=float), fourier_df.to_numpy(dtype=float)]
    )

    for k in range(len(cuts) - 1):
        cut_k, cut_next = cuts[k], cuts[k + 1]
        train_y = y_all[:cut_k]
        train_X = X_all[:cut_k]

        model = ARIMA(
            order=model_params["order"],
            season_length=model_params["season_length"],
            seasonal_order=model_params["seasonal_order"],
            method=method,
        )
        model.fit(train_y, train_X)
        coefs = model.model_.get("coef", {})
        if not coefs or not all(np.isfinite(float(v)) for v in coefs.values()):
            fold_results.append({
                "fold": k + 1,
                "train_until": str(target.index[cut_k - 1]),
                "eval_from": str(target.index[cut_k]),
                "eval_to": str(target.index[cut_next - 1]),
                "n_windows": 0,
                "rmse": float("nan"),
            })
            del model
            gc.collect()
            continue
        fixed = {kk: float(v) for kk, v in coefs.items()}
        del model
        gc.collect()

        rmses = []
        abs_pos = cut_k
        while abs_pos + horizon <= cut_next:
            act = y_all[abs_pos:abs_pos + horizon]
            pred = _fixed_window_forecast(
                y_all, X_all, abs_pos, context_window, horizon,
                model_params["order"], model_params["seasonal_order"],
                model_params["season_length"], fixed, method,
            )
            rmses.append(float(np.sqrt(np.mean((act - pred) ** 2))))
            all_actual.append(act)
            all_pred.append(pred)
            abs_pos += stride

        fold_rmse = float(np.mean(rmses)) if rmses else float("nan")
        fold_results.append({
            "fold": k + 1,
            "train_until": str(target.index[cut_k - 1]),
            "eval_from": str(target.index[cut_k]),
            "eval_to": str(target.index[cut_next - 1]),
            "n_windows": len(rmses),
            "rmse": fold_rmse,
        })

    valid = [r["rmse"] for r in fold_results if not np.isnan(r["rmse"])]
    cv_rmse = float(np.mean(valid)) if valid else float("inf")
    pooled = compute_metrics(np.concatenate(all_actual), np.concatenate(all_pred)) if all_actual else None
    return fold_results, cv_rmse, pooled

#IMPORTANTE: el test se evalua una sola vez; no ajustar con esos datos.
def evaluate_test(target, covariates, fourier_periods, model_params, test_start_idx, horizon, stride, context_window, method):
    """Refit con train+val completo y evalúa test UNA sola vez (holdout)."""
    y_all = target.to_numpy(dtype=float)
    fourier_df = _build_fourier_df(
        target.index, fourier_periods,
        model_params["k_daily"], model_params["k_weekly"], model_params["k_yearly"],
    )
    cov_features = [f for f in model_params["features"] if f in covariates.columns]
    X_all = np.hstack(
        [covariates[cov_features].to_numpy(dtype=float), fourier_df.to_numpy(dtype=float)]
    )

    trainval_y = y_all[:test_start_idx]
    trainval_X = X_all[:test_start_idx]

    model = ARIMA(
        order=model_params["order"],
        season_length=model_params["season_length"],
        seasonal_order=model_params["seasonal_order"],
        method=method,
    )
    model.fit(trainval_y, trainval_X)
    fixed = {kk: float(v) for kk, v in model.model_.get("coef", {}).items()}
    del model
    gc.collect()

    all_actual, all_pred = [], []
    abs_pos = test_start_idx
    while abs_pos + horizon <= len(target):
        all_actual.append(y_all[abs_pos:abs_pos + horizon])
        all_pred.append(_fixed_window_forecast(
            y_all, X_all, abs_pos, context_window, horizon,
            model_params["order"], model_params["seasonal_order"],
            model_params["season_length"], fixed, method,
        ))
        abs_pos += stride

    if not all_actual:
        return None, 0, np.array([]), np.array([])
    metrics = compute_metrics(np.concatenate(all_actual), np.concatenate(all_pred))
    return metrics, len(all_actual), np.concatenate(all_actual), np.concatenate(all_pred)

def main():
    config = load_config(CONFIG_PATH)

    LOG_DIR   = Path(config.get("output", {}).get("logs", "output/logs"))
    EVAL_DIR  = Path(config.get("output", {}).get("evaluation", "output/evaluation"))
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("sarimax_evaluation")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_DIR / "sarimax_evaluation.log")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    dataset_cfg = config.get("dataset", {})
    csv_path    = dataset_cfg.get("input_file_sarimax")
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

    target = df[target_col]

    cov_cols = [
        c for c in df.columns
        if c not in [date_col, hour_col, target_col] and not c.startswith("fourier_")
    ]
    covariates = df[cov_cols]

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

    n_total        = len(target)
    val_start_idx  = int(target.index.searchsorted(VAL_START))
    test_start_idx = int(target.index.searchsorted(TEST_START))

    bt_cfg           = config.get("backtesting", {})
    FORECAST_HORIZON = bt_cfg.get("forecast_horizon", 24)
    STRIDE           = bt_cfg.get("stride", 24)
    CONTEXT_WINDOW   = bt_cfg.get("context_window", 24 * 7)

    sarimax_cfg = config.get("sarimax", {})
    METHOD = sarimax_cfg.get("method", "CSS")
    search_cfg = sarimax_cfg.get("search", {})
    fourier_cfg = sarimax_cfg.get("fourier", {})
    fourier_periods = fourier_cfg.get("periods", {"daily": 24.0, "weekly": 168.0, "yearly": 8765.82})

    fixed_features = []
    fixed_cfg = sarimax_cfg.get("features", {}).get("fixed", {})
    if isinstance(fixed_cfg, dict):
        for group in fixed_cfg.values():
            if isinstance(group, dict) and "features" in group:
                fixed_features.extend(group["features"])
            elif isinstance(group, list):
                fixed_features.extend(group)
    elif isinstance(fixed_cfg, list):
        fixed_features.extend(fixed_cfg)

    candidates_cfg = sarimax_cfg.get("features", {}).get("candidates_group", {})

    n_train = len(target[target.index < VAL_START])
    n_val   = len(target[(target.index >= VAL_START) & (target.index < TEST_START)])
    n_test  = len(target[target.index >= TEST_START])

    print(f"Train: {n_train} obs (hasta {VAL_START})")
    print(f"Val:   {n_val} obs ({VAL_START} -> {TEST_START})")
    print(f"Test:  {n_test} obs (desde {TEST_START}, reservado)")
    print(f"Walk-forward: horizon={FORECAST_HORIZON}h, stride={STRIDE}h, context_window={CONTEXT_WINDOW}h")

    cuts = build_cv_cuts_idx(test_start_idx, N_FOLDS, MIN_TRAIN_HOURS)
    print(f"\nCV temporal ({N_FOLDS} folds, expanding window, dentro de train+val):")
    for k in range(len(cuts) - 1):
        print(f"  fold {k + 1}: fit < {target.index[cuts[k]]}  |  eval [{target.index[cuts[k]]}, {target.index[cuts[k + 1] - 1]})")

    if not TRIALS_CSV.exists():
        print(f"\nError: no se encontró {TRIALS_CSV}. Corré primero src/optimization/sarimax_optuna.py")
        sys.exit(1)

    trials_df = pd.read_csv(TRIALS_CSV)
    trials_df["val_rmse"] = pd.to_numeric(trials_df["user_attrs_val_rmse"], errors="coerce")

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
        test_rmse = r.get("user_attrs_test_rmse")
        test_rmse_s = "?" if pd.isna(test_rmse) else f"{float(test_rmse):8.4f}"
        print(f"  trial #{int(r['number']):>3d}  val_rmse={float(r['val_rmse']):8.4f}  "
              f"test_rmse(orientativo)={test_rmse_s}")

    print(f"\nCorriendo CV temporal para {N_CAND} candidatos x {N_FOLDS} folds...")
    print("=" * 80)

    results = []
    for _, row in candidates.iterrows():
        trial_num = int(row["number"])
        model_params = build_model_params(row, fixed_features, candidates_cfg, search_cfg)
        t0 = time.time()
        fold_results, cv_rmse, pooled = run_cv(
            target, covariates, fourier_periods, model_params, cuts,
            FORECAST_HORIZON, STRIDE, CONTEXT_WINDOW, METHOD,
        )
        elapsed = time.time() - t0

        fold_txt = " | ".join(
            f"f{r['fold']}={r['rmse']:.2f}({r['n_windows']}w)" for r in fold_results
        )
        print(f"[trial {trial_num:>3d}] CV_RMSE={cv_rmse:8.4f}  ({elapsed:5.0f}s)  {fold_txt}")

        selected_candidate = []
        if isinstance(row.get("user_attrs_selected_candidate_features"), str):
            try:
                selected_candidate = ast.literal_eval(row["user_attrs_selected_candidate_features"])
            except Exception:
                selected_candidate = []

        results.append({
            "trial_number": trial_num,
            "val_rmse": float(row["val_rmse"]),
            "cv_rmse": cv_rmse,
            "cv_rmse_pooled": pooled["rmse"] if pooled else None,
            "cv_mae":  pooled["mae"]  if pooled else None,
            "cv_mape": pooled["mape"] if pooled else None,
            "cv_smape": pooled["smape"] if pooled else None,
            "cv_r2":   pooled["r2"]   if pooled else None,
            "p": model_params["order"][0],
            "d": model_params["order"][1],
            "q": model_params["order"][2],
            "P": model_params["seasonal_order"][0],
            "D": model_params["seasonal_order"][1],
            "Q": model_params["seasonal_order"][2],
            "seasonal_period": model_params["season_length"],
            "k_daily": model_params["k_daily"],
            "k_weekly": model_params["k_weekly"],
            "k_yearly": model_params["k_yearly"],
            "selected_features": selected_candidate,
            "cv_seconds": round(elapsed, 1),
        })

    results_df = pd.DataFrame(results).sort_values("cv_rmse").reset_index(drop=True)
    horizon = config.get("backtesting", {}).get("forecast_horizon", 24)
    suffix = f"_{horizon}h"
    cv_csv_name = f"sarimax_cv_top10{suffix}.csv"
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
    winner_row = candidates[candidates["number"] == winner_trial].iloc[0]
    model_params = build_model_params(winner_row, fixed_features, candidates_cfg, search_cfg)

    print("\n" + "=" * 80)
    print(f"EVALUACIÓN FINAL SOBRE TEST (holdout, UNA sola vez)")
    print(f"Ganador por CV: trial #{winner_trial}")
    print("=" * 80)

    test_metrics, n_windows_test, test_actual, test_pred = evaluate_test(
        target, covariates, fourier_periods, model_params, test_start_idx,
        FORECAST_HORIZON, STRIDE, CONTEXT_WINDOW, METHOD,
    )

    if test_metrics:
        print(f"\n  Métricas en test ({n_windows_test} ventanas):")
        for k, v in test_metrics.items():
            print(f"    {k:6s} = {v:.4f}")
        logger.info(f"Test metrics trial #{winner_trial}: {test_metrics}")

        horizon = config.get("backtesting", {}).get("forecast_horizon", 24)
        n = len(test_actual)
        test_datetime = []
        abs_pos = test_start_idx
        while abs_pos + FORECAST_HORIZON <= len(target) and len(test_datetime) < n:
            test_datetime.extend(target.index[abs_pos:abs_pos + FORECAST_HORIZON].tolist())
            abs_pos += STRIDE
        pred_df = pd.DataFrame({
            "datetime": test_datetime[:n],
            "actual": test_actual,
            "predicted": test_pred,
        })
        pred_csv = EVAL_DIR / f"sarimax_test_predictions_{horizon}h.csv"
        pred_df.to_csv(pred_csv, index=False)

        write_forecast_meta(
            pred_csv, model="sarimax", horizon=FORECAST_HORIZON, stride=STRIDE,
            context_window=CONTEXT_WINDOW, test_start=TEST_START,
            n_windows=int(n_windows_test), n_predictions=n,
        )
        print(f"  Predicciones del test guardadas en: {pred_csv}")
    else:
        print("\n  No se generaron ventanas de test.")

    winner_info = {
        "trial_number": winner_trial,
        "cv_rmse": float(winner["cv_rmse"]),
        "val_rmse": float(winner["val_rmse"]),
        "hyperparameters": {
            "order": (int(winner["p"]), int(winner["d"]), int(winner["q"])),
            "seasonal_order": (int(winner["P"]), int(winner["D"]), int(winner["Q"]), int(winner["seasonal_period"])),
            "seasonal_period": int(winner["seasonal_period"]),
            "k_daily": int(winner["k_daily"]),
            "k_weekly": int(winner["k_weekly"]),
            "k_yearly": int(winner["k_yearly"]),
        },
        "fixed_features": fixed_features,
        "selected_features": winner["selected_features"],
        "test_metrics": test_metrics,
        "n_windows_test": n_windows_test,
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_yaml = EVAL_DIR / f"sarimax_cv_winner_{horizon}h.yaml"
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml_rt.dump(winner_info, f)
    print(f"\n  Detalle del ganador guardado en: {out_yaml}")

    horizon = config.get("backtesting", {}).get("forecast_horizon", 24)
    horizon_key = f"best_{horizon}h"
    if "sarimax" not in config:
        config["sarimax"] = {}

    new_best = {}
    new_best["p"] = int(winner["p"])
    new_best["d"] = int(winner["d"])
    new_best["q"] = int(winner["q"])
    new_best["P"] = int(winner["P"])
    new_best["D"] = int(winner["D"])
    new_best["Q"] = int(winner["Q"])
    new_best["seasonal_period"] = int(winner["seasonal_period"])
    new_best["k_daily"]  = int(winner["k_daily"])
    new_best["k_weekly"] = int(winner["k_weekly"])
    new_best["k_yearly"] = int(winner["k_yearly"])
    new_best["features"] = {
        "fixed": fixed_features,
        "selected": list(winner["selected_features"]),
    }
    new_best["rmse"] = _round_metric(winner["val_rmse"])

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

    new_best["acf_24"]  = row_to_float(winner_row.get("user_attrs_acf_24"))
    new_best["acf_168"] = row_to_float(winner_row.get("user_attrs_acf_168"))
    new_best["aic"]     = row_to_float(winner_row.get("user_attrs_aic"))
    new_best["bic"]     = row_to_float(winner_row.get("user_attrs_bic"))

    new_best["trial_number"] = int(winner_trial)
    new_best["datetime"]     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    config["sarimax"][horizon_key] = new_best
    save_config(config, CONFIG_PATH)
    print(f"\n  Ganador escrito en 'sarimax.{horizon_key}' en {CONFIG_PATH}")
    logger.info(f"Ganador trial #{winner_trial} escrito en sarimax.{horizon_key}")

    print("\nNota: el test_rmse reportado es el error real del modelo elegido por CV.")

if __name__ == "__main__":
    main()