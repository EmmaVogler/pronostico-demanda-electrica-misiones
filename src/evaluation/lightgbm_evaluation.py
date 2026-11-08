import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import re
import sys
import gc
import time
import logging
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from ruamel.yaml import YAML

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast_meta import write_forecast_meta  
from darts import TimeSeries
from darts.models import LightGBMModel

warnings.filterwarnings("ignore")

CONFIG_PATH = Path("config.yaml")
TRIALS_CSV  = Path("output/optimization/lightgbm_trials.csv")

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

def _yamlable(obj):
    """Convierte escalares de numpy a tipos de Python antes de serializar.

    ruamel.yaml no tiene representer para np.float64/np.int64/np.bool_ y tumba
    el dump con `RepresenterError: cannot represent an object: np.float64(nan)`.
    Un nan/inf de Python sí se serializa (.nan/.inf), pero en reportes y en la
    configuración es más claro un null, así que los no finitos van a None.
    """
    if isinstance(obj, dict):
        return {k: _yamlable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_yamlable(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj

def _group_names(value) -> list:
    """Nombres de grupo a partir de la celda del CSV de trials.

    pandas lee las celdas vacías de `*_groups_selected` como NaN, y ese nan de
    numpy era el que llegaba al YAML del ganador (y además se serializaba como
    el string "nan" al escribir la configuración). Vacío/None/NaN devuelve la
    lista vacía.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [g for g in str(value).split(",") if g]

for _np_cls in (np.float64, np.float32, np.float16,
                np.int64, np.int32, np.int16, np.int8,
                np.uint64, np.uint32, np.uint16, np.uint8, np.bool_):
    yaml_rt.Representer.add_representer(_np_cls, lambda y, d: y.represent_data(d.item()))

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

def load_lightgbm_features(config: dict) -> dict:
    """Extrae la config de features desde config.yaml -> lightgbm.features."""
    feats = config.get("lightgbm", {}).get("features", {})

    past_covariates   = feats.get("past_covariates") or []
    future_covariates = feats.get("future_covariates") or []
    past_groups       = feats.get("past_groups") or {}
    future_groups     = feats.get("future_groups") or {}
    lags_target       = feats.get("lags_target", [-1, -24, -168])
    lags_past_covs    = feats.get("lags_past_covs")
    lags_future_covs  = feats.get("lags_future_covs", [0])

    if not past_covariates and not future_covariates:
        raise ValueError(
            "lightgbm.features.past_covariates / future_covariates no pueden estar vacíos ambos"
        )
    if past_covariates and lags_past_covs is None:
        raise ValueError(
            "lightgbm.features.lags_past_covs es obligatorio si hay past_covariates"
        )

    return {
        "past_covariates":   past_covariates,
        "future_covariates": future_covariates,
        "past_groups":       past_groups,
        "future_groups":     future_groups,
        "lags_target":       lags_target,
        "lags_past_covs":    lags_past_covs,
        "lags_future_covs":  lags_future_covs,
    }

FOURIER_COL_RE = re.compile(r"^fourier_(daily|weekly|yearly)_(sin|cos)_(\d+)$")
FOURIER_PERIODS = ("daily", "weekly", "yearly")

def clip_fourier_columns(cols, period: str, k) -> list:
    """Recorta un grupo de armónicos de Fourier a los primeros k términos.

    El grupo `use_fourier_<period>` se activa completo o nada, pero cada trial
    registra cuántos armónicos utilizó (`fourier_k_<period>` en el CSV de
    trials). Con k = 0 el grupo queda sin columnas; con k menor que el tamaño
    del grupo se conservan solo los armónicos 1..k, y con k mayor se conserva
    todo. Si el CSV no trae la columna de k (corridas anteriores al registro),
    se devuelve el grupo completo.
    """
    if k is None or pd.isna(k):
        return list(cols)
    k = int(k)
    if k <= 0:
        return []
    keep = []
    for col in cols:
        match = FOURIER_COL_RE.match(str(col))
        if match and match.group(1) == period and int(match.group(3)) > k:
            continue
        keep.append(col)
    return keep

def select_groups(params_row, candidate_groups: dict):
    """Grupos activados por un trial y sus columnas efectivas.

    Devuelve (grupos, columnas). Las columnas de Fourier se recortan al k
    registrado en el trial; el resto de grupos se toma completo.
    """
    selected, columns = [], []
    for name, cols in candidate_groups.items():
        if not bool(params_row.get(name)):
            continue
        selected.append(name)
        if name.startswith("use_fourier_"):
            period = name[len("use_fourier_"):]
            columns.extend(clip_fourier_columns(cols, period, params_row.get(f"fourier_k_{period}")))
        else:
            columns.extend(cols)
    return selected, columns

def fourier_k(cols) -> dict:
    """Número de armónicos de Fourier presente en una lista de columnas."""
    ks = {period: 0 for period in FOURIER_PERIODS}
    for col in cols:
        match = FOURIER_COL_RE.match(str(col))
        if match:
            period, k = match.group(1), int(match.group(3))
            ks[period] = max(ks[period], k)
    return ks

def build_model(params_row, candidate_past_groups, candidate_future_groups,
                lags_target, lags_past_covs, lags_future_covs, seed,
                output_chunk_length=1):
    """Construye un LightGBMModel a partir de una fila del CSV de trials.

    Devuelve (model, past_cols_eff, future_cols_eff).

    output_chunk_length = 1: el modelo se ajusta como pronóstico de un paso y
    Darts lo itera hasta cubrir el horizonte (esquema recursivo), igual que en
    lightgbm_optuna.py. Antes la evaluación usaba output_chunk_length =
    forecast_horizon (modelo directo de 24 pasos), con lo que se evaluaba un
    modelo distinto al que había optimizado Optuna: sobre el mismo trial, el
    RMSE de la CV pasaba de ~34 a ~43 y el de la ventana de validación de ~24
    a ~38. Los hiperparámetros de la búsqueda solo son válidos para el modelo
    recursivo, por lo que ambos scripts deben usar el mismo esquema.
    """
    _, past_cols_eff   = select_groups(params_row, candidate_past_groups)
    _, future_cols_eff = select_groups(params_row, candidate_future_groups)

    model = LightGBMModel(
        lags=lags_target,
        lags_past_covariates=lags_past_covs if past_cols_eff else None,
        lags_future_covariates=lags_future_covs if future_cols_eff else None,
        output_chunk_length=int(output_chunk_length),
        learning_rate=float(params_row["learning_rate"]),
        num_leaves=int(params_row["num_leaves"]),
        max_depth=int(params_row["max_depth"]),
        min_child_samples=int(params_row["min_child_samples"]),
        feature_fraction=float(params_row["feature_fraction"]),
        bagging_fraction=float(params_row["bagging_fraction"]),
        lambda_l1=float(params_row["lambda_l1"]),
        lambda_l2=float(params_row["lambda_l2"]),
        n_estimators=int(params_row["n_estimators"]),
        random_state=seed,
        verbose=-1,
    )
    return model, past_cols_eff, future_cols_eff

def make_factory(params_row, candidate_past_groups, candidate_future_groups,
                 lags_target, lags_past_covs, lags_future_covs, seed,
                 output_chunk_length=1):
    """Devuelve una función que construye un modelo fresco (para cada fold)."""
    def factory():
        return build_model(params_row, candidate_past_groups, candidate_future_groups,
                           lags_target, lags_past_covs, lags_future_covs, seed,
                           output_chunk_length)
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

def run_cv(series, past_covs, future_covs, model_factory, cuts, horizon, stride):
    """Expanding-window CV temporal dentro de train+val.

    fold k: fit con datos < cuts[k], walk-forward predict en [cuts[k], cuts[k+1]).
    Devuelve (fold_results, cv_rmse, pooled_metrics).
    cv_rmse = media de RMSE por ventana (mismo criterio que Optuna).
    """
    fold_results = []
    all_actual, all_pred = [], []

    for k in range(len(cuts) - 1):
        cut_k, cut_next = cuts[k], cuts[k + 1]
        model, past_cols, future_cols = model_factory()

        train_series = series.drop_after(cut_k)
        fit_kwargs = {"series": train_series, "verbose": False}
        if past_cols:
            fit_kwargs["past_covariates"] = past_covs.drop_after(cut_k)[past_cols]
        if future_cols:
            fit_kwargs["future_covariates"] = future_covs.drop_after(cut_k)[future_cols]
        model.fit(**fit_kwargs)

        hf_kwargs = {
            "series": series,
            "start": cut_k,
            "forecast_horizon": horizon,
            "stride": stride,
            "retrain": False,
            "overlap_end": False,
            "last_points_only": False,
            "verbose": False,
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

    valid = [r["rmse"] for r in fold_results if not np.isnan(r["rmse"])]
    cv_rmse = float(np.mean(valid)) if valid else float("inf")
    pooled = compute_metrics(np.concatenate(all_actual), np.concatenate(all_pred)) if all_actual else None
    return fold_results, cv_rmse, pooled

def evaluate_test(series, past_covs, future_covs, model_factory, test_start, horizon, stride):
    """Refit con train+val completo y evalúa test UNA sola vez (holdout)."""
    model, past_cols, future_cols = model_factory()

    trainval_series = series.drop_after(test_start)
    fit_kwargs = {"series": trainval_series, "verbose": False}
    if past_cols:
        fit_kwargs["past_covariates"] = past_covs.drop_after(test_start)[past_cols]
    if future_cols:
        fit_kwargs["future_covariates"] = future_covs.drop_after(test_start)[future_cols]
    model.fit(**fit_kwargs)

    hf_kwargs = {
        "series": series,
        "start": test_start,
        "forecast_horizon": horizon,
        "stride": stride,
        "retrain": False,
        "overlap_end": False,
        "last_points_only": False,
        "verbose": False,
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
    np.random.seed(seed)

    feats = load_lightgbm_features(config)
    past_covariates         = feats["past_covariates"]
    future_covariates       = feats["future_covariates"]
    candidate_past_groups   = feats["past_groups"]
    candidate_future_groups = feats["future_groups"]
    lags_target             = feats["lags_target"]
    lags_past_covs          = feats["lags_past_covs"]
    lags_future_covs        = feats["lags_future_covs"]

    LOG_DIR   = Path(config.get("output", {}).get("logs", "output/logs"))
    EVAL_DIR  = Path(config.get("output", {}).get("evaluation", "output/evaluation"))
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("lightgbm_evaluation")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_DIR / "lightgbm_evaluation.log")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    dataset_cfg = config.get("dataset", {})
    csv_path    = dataset_cfg.get("input_file_lightgbm")
    date_col    = dataset_cfg.get("datetime", {}).get("date_column", "FECHA")
    hour_col    = dataset_cfg.get("datetime", {}).get("hour_column", "Horario")
    target_col  = dataset_cfg.get("target", "TOTAL_SIP")
    period_cfg  = dataset_cfg.get("period", {})

    print(f"Cargando datos desde {csv_path}...")
    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(df[date_col]) + pd.to_timedelta(df[hour_col], unit="h")
    df = df.set_index("datetime").sort_index()

    period_start = pd.to_datetime(str(period_cfg.get("start", "2023-01-01")))
    period_end   = pd.to_datetime(str(period_cfg.get("end", "2026-01-01")))

    split_cfg   = config.get("split", {})
    train_ratio = split_cfg.get("train_ratio", 0.70)
    val_ratio   = split_cfg.get("validation_ratio", 0.15)

    df = df.loc[(df.index >= period_start) & (df.index <= period_end)]

    period_hours  = (period_end - period_start).total_seconds() / 3600
    train_hours   = int(period_hours * train_ratio)
    val_hours     = int(period_hours * val_ratio)

    VAL_START  = period_start + pd.Timedelta(hours=train_hours)
    TEST_START = period_start + pd.Timedelta(hours=train_hours + val_hours)
    VAL_END    = TEST_START

    n_total        = len(df)
    val_start_idx  = int(df.index.searchsorted(VAL_START))
    test_start_idx = int(df.index.searchsorted(TEST_START))

    n_train = val_start_idx
    n_val   = test_start_idx - val_start_idx
    n_test  = n_total - test_start_idx

    bt_cfg           = config.get("backtesting", {})
    FORECAST_HORIZON = bt_cfg.get("forecast_horizon", 24)
    STRIDE           = bt_cfg.get("stride", 24)

    print(f"Train: {n_train} obs (hasta {VAL_START})")
    print(f"Val:   {n_val} obs ({VAL_START} -> {VAL_END})")
    print(f"Test:  {n_test} obs (desde {TEST_START}, reservado)")
    print(f"Walk-forward: horizon={FORECAST_HORIZON}h, stride={STRIDE}h")

    series_ts      = TimeSeries.from_dataframe(df[[target_col]], freq="h")
    past_covs_ts   = TimeSeries.from_dataframe(df[past_covariates], freq="h")
    future_covs_ts = TimeSeries.from_dataframe(df[future_covariates], freq="h")

    cuts = build_cv_cuts(period_start, TEST_START, N_FOLDS, MIN_TRAIN_HOURS)
    print(f"\nCV temporal ({N_FOLDS} folds, expanding window, dentro de train+val):")
    for k in range(len(cuts) - 1):
        print(f"  fold {k + 1}: fit < {cuts[k]}  |  eval [{cuts[k]}, {cuts[k + 1]})")

    if not TRIALS_CSV.exists():
        print(f"\nError: no se encontró {TRIALS_CSV}. Corré primero src/optimization/lightgbm_optuna.py")
        sys.exit(1)

    trials_df = pd.read_csv(TRIALS_CSV)
    trials_df["val_rmse"]      = pd.to_numeric(trials_df["val_rmse"], errors="coerce")
    trials_df["n_estimators"]  = pd.to_numeric(trials_df["n_estimators"], errors="coerce")

    valid = trials_df.dropna(subset=["val_rmse", "n_estimators"])
    valid = valid[valid["val_rmse"] != float("inf")]

    if len(valid) < N_CANDIDATES:
        print(f"\nAdvertencia: solo hay {len(valid)} trials válidos (se pidieron {N_CANDIDATES}).")
        N_CAND = len(valid)
    else:
        N_CAND = N_CANDIDATES

    candidates = valid.nsmallest(N_CAND, "val_rmse")

    print(f"\n=== CANDIDATOS (top-{N_CAND} por val_rmse) ===")
    for _, r in candidates.iterrows():
        _, r_past   = select_groups(r, candidate_past_groups)
        _, r_future = select_groups(r, candidate_future_groups)
        k = fourier_k(r_past + r_future)
        k_txt = f"{k['daily']}/{k['weekly']}/{k['yearly']}"
        print(f"  trial #{int(r['trial_number']):>3d}  val_rmse={float(r['val_rmse']):8.4f}  "
              f"test_rmse(orientativo)={float(r['test_rmse']):8.4f}  k_fourier={k_txt}")

    print(f"\nCorriendo CV temporal para {N_CAND} candidatos x {N_FOLDS} folds...")
    print("=" * 80)

    results = []
    for _, row in candidates.iterrows():
        trial_num = int(row["trial_number"])
        _, cand_past   = select_groups(row, candidate_past_groups)
        _, cand_future = select_groups(row, candidate_future_groups)
        k = fourier_k(cand_past + cand_future)
        factory = make_factory(row, candidate_past_groups, candidate_future_groups,
                               lags_target, lags_past_covs, lags_future_covs, seed,
                               output_chunk_length=1)
        t0 = time.time()
        fold_results, cv_rmse, pooled = run_cv(
            series_ts, past_covs_ts, future_covs_ts, factory, cuts,
            FORECAST_HORIZON, STRIDE,
        )
        elapsed = time.time() - t0

        fold_txt = " | ".join(
            f"f{r['fold']}={r['rmse']:.2f}({r['n_windows']}w)" for r in fold_results
        )
        print(f"[trial {trial_num:>3d}] CV_RMSE={cv_rmse:8.4f}  ({elapsed:5.0f}s)  "
              f"k_fourier={k['daily']}/{k['weekly']}/{k['yearly']}  {fold_txt}")

        results.append({
            "trial_number": trial_num,
            "val_rmse": float(row["val_rmse"]),
            "cv_rmse": cv_rmse,
            "cv_rmse_pooled": pooled["rmse"] if pooled else None,
            "cv_mae":  pooled["mae"]  if pooled else None,
            "cv_mape": pooled["mape"] if pooled else None,
            "cv_smape": pooled["smape"] if pooled else None,
            "cv_r2":   pooled["r2"]   if pooled else None,
            "learning_rate": float(row["learning_rate"]),
            "num_leaves": int(row["num_leaves"]),
            "max_depth": int(row["max_depth"]),
            "min_child_samples": int(row["min_child_samples"]),
            "feature_fraction": float(row["feature_fraction"]),
            "bagging_fraction": float(row["bagging_fraction"]),
            "lambda_l1": float(row["lambda_l1"]),
            "lambda_l2": float(row["lambda_l2"]),
            "n_estimators": int(row["n_estimators"]),
            "past_groups": row["past_groups_selected"],
            "future_groups": row["future_groups_selected"],
            "fourier_k_daily": k["daily"],
            "fourier_k_weekly": k["weekly"],
            "fourier_k_yearly": k["yearly"],
            "cv_seconds": round(elapsed, 1),
        })

    results_df = pd.DataFrame(results).sort_values("cv_rmse").reset_index(drop=True)
    horizon = config.get("backtesting", {}).get("forecast_horizon", 24)
    suffix = f"_{horizon}h"
    cv_csv_name = f"lightgbm_cv_top10{suffix}.csv"
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
    factory = make_factory(winner_row, candidate_past_groups, candidate_future_groups,
                           lags_target, lags_past_covs, lags_future_covs, seed,
                           output_chunk_length=1)

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
        pred_csv = EVAL_DIR / f"lightgbm_test_predictions_{horizon}h.csv"
        pred_df.to_csv(pred_csv, index=False)

        write_forecast_meta(
            pred_csv, model="lightgbm", horizon=FORECAST_HORIZON, stride=STRIDE,
            context_window=None, test_start=TEST_START,
            n_windows=int(n_windows_test), n_predictions=len(pred_df),
        )
        print(f"  Predicciones del test guardadas en: {pred_csv}")
    else:
        print("  No se generaron ventanas de test.")
        logger.warning("No se generaron ventanas de test")

    winner_info = _yamlable({
        "trial_number": winner_trial,
        "cv_rmse": float(winner["cv_rmse"]),
        "val_rmse": float(winner["val_rmse"]),
        "hyperparameters": {
            "learning_rate": float(winner["learning_rate"]),
            "num_leaves": int(winner["num_leaves"]),
            "max_depth": int(winner["max_depth"]),
            "min_child_samples": int(winner["min_child_samples"]),
            "feature_fraction": float(winner["feature_fraction"]),
            "bagging_fraction": float(winner["bagging_fraction"]),
            "lambda_l1": float(winner["lambda_l1"]),
            "lambda_l2": float(winner["lambda_l2"]),
            "n_estimators": int(winner["n_estimators"]),
        },
        "past_groups": _group_names(winner["past_groups"]),
        "future_groups": _group_names(winner["future_groups"]),
        "test_metrics": test_metrics,
        "n_windows_test": n_windows_test,
        "output_chunk_length": 1,
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    out_yaml = EVAL_DIR / f"lightgbm_cv_winner_{horizon}h.yaml"
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml_rt.dump(winner_info, f)
    print(f"\n  Detalle del ganador guardado en: {out_yaml}")

    horizon = config.get("backtesting", {}).get("forecast_horizon", 24)
    horizon_key = f"best_{horizon}h"
    if "lightgbm" not in config:
        config["lightgbm"] = {}

    new_best = {}
    new_best["learning_rate"]      = float(winner["learning_rate"])
    new_best["num_leaves"]         = int(winner["num_leaves"])
    new_best["max_depth"]          = int(winner["max_depth"])
    new_best["min_child_samples"]  = int(winner["min_child_samples"])
    new_best["feature_fraction"]   = float(round(float(winner["feature_fraction"]), 4))
    new_best["bagging_fraction"]   = float(round(float(winner["bagging_fraction"]), 4))
    new_best["lambda_l1"]          = float(round(float(winner["lambda_l1"]), 4))
    new_best["lambda_l2"]          = float(round(float(winner["lambda_l2"]), 4))
    new_best["n_estimators"]       = int(winner["n_estimators"])
    new_best["lags"]               = lags_target
    new_best["lags_past_covariates"]     = lags_past_covs if winner["past_groups"] else None
    new_best["lags_future_covariates"]   = lags_future_covs if winner["future_groups"] else None
    new_best["past_covariates"]    = [col for g in _group_names(winner["past_groups"])   for col in candidate_past_groups.get(g, [])]
    new_best["future_covariates"]  = [col for g in _group_names(winner["future_groups"]) for col in candidate_future_groups.get(g, [])]
    new_best["past_groups"]        = _group_names(winner["past_groups"])
    new_best["future_groups"]      = _group_names(winner["future_groups"])
    new_best["fourier_k_daily"]    = int(winner.get("fourier_k_daily") or 0)
    new_best["fourier_k_weekly"]   = int(winner.get("fourier_k_weekly") or 0)
    new_best["fourier_k_yearly"]   = int(winner.get("fourier_k_yearly") or 0)
    new_best["output_chunk_length"] = 1

    def _round_metric(v):
        """Redondea a 4 decimales; None/NaN se guardan como None."""
        if v is None:
            return None
        v = float(v)
        return float(round(v, 4)) if not np.isnan(v) else None

    new_best["cv_rmse"]            = _round_metric(winner["cv_rmse"])
    new_best["cv_rmse_pooled"]     = _round_metric(winner["cv_rmse_pooled"])
    new_best["cv_mae"]             = _round_metric(winner["cv_mae"])
    new_best["cv_mape"]            = _round_metric(winner["cv_mape"])
    new_best["cv_smape"]           = _round_metric(winner["cv_smape"])
    new_best["cv_r2"]              = _round_metric(winner["cv_r2"])

    new_best["rmse_test"]          = _round_metric(test_metrics["rmse"]  if test_metrics else None)
    new_best["mae_test"]           = _round_metric(test_metrics["mae"]   if test_metrics else None)
    new_best["mape_test"]          = _round_metric(test_metrics["mape"]  if test_metrics else None)
    new_best["smape_test"]         = _round_metric(test_metrics["smape"] if test_metrics else None)
    new_best["r2_test"]            = _round_metric(test_metrics["r2"]    if test_metrics else None)
    new_best["trial_number"]       = int(winner_trial)
    new_best["datetime"]           = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    config["lightgbm"][horizon_key] = new_best
    save_config(config, CONFIG_PATH)
    print(f"\n  Ganador escrito en 'lightgbm.{horizon_key}' en {CONFIG_PATH}")
    logger.info(f"Ganador trial #{winner_trial} escrito en lightgbm.{horizon_key}")

    print("\nNota: el test_rmse reportado es el error real del modelo elegido por CV.")

if __name__ == "__main__":
    main()