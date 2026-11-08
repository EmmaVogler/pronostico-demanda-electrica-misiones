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
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import optuna
import torch
torch.backends.fp32_precision = "ieee"

from statsmodels.tsa.stattools import acf
from ruamel.yaml import YAML
from darts import TimeSeries
from darts.models import Chronos2Model

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

CONFIG_PATH = Path("config.yaml")

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

HUB_MODEL_VARIANTS = [
    "amazon/chronos-2"
]

ALL_PAST_COLS = list(dict.fromkeys(
    c for cols in CANDIDATE_PAST_GROUPS.values() for c in cols
))
ALL_FUTURE_COLS = list(dict.fromkeys(
    c for cols in CANDIDATE_FUTURE_GROUPS.values() for c in cols
))

def build_objective(
    series: TimeSeries,
    past_covs_full: TimeSeries,
    future_covs_full: TimeSeries,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
    forecast_horizon: int,
    stride: int,
    context_window: int,
    chronos_search_cfg: dict,
    device: str,
):
    model_variants = chronos_search_cfg.get("hub_model_name", HUB_MODEL_VARIANTS)

    def objective(trial: optuna.Trial) -> float:
        hub_model_name = trial.suggest_categorical("hub_model_name", model_variants)

        selected_past_cols = []
        for flag, cols in CANDIDATE_PAST_GROUPS.items():
            if trial.suggest_categorical(flag, [True, False]):
                selected_past_cols.extend(cols)
        selected_past_cols = list(dict.fromkeys(selected_past_cols))

        selected_future_cols = []
        for flag, cols in CANDIDATE_FUTURE_GROUPS.items():
            if trial.suggest_categorical(flag, [True, False]):
                selected_future_cols.extend(cols)
        selected_future_cols = list(dict.fromkeys(selected_future_cols))

        try:
            t_fit_start = time.time()

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

            past_covs   = past_covs_full[selected_past_cols] if past_covs_full and selected_past_cols else None
            future_covs = future_covs_full[selected_future_cols] if future_covs_full and selected_future_cols else None

            train_series      = series.drop_after(val_start)
            train_past_covs   = past_covs.drop_after(val_start) if past_covs is not None else None
            train_future_covs = future_covs.drop_after(val_start) if future_covs is not None else None

            model.fit(
                series=train_series,
                past_covariates=train_past_covs,
                future_covariates=train_future_covs,
                verbose=False,
            )
            fit_seconds = time.time() - t_fit_start

            t_wf_start = time.time()
            backtest_forecasts = model.historical_forecasts(
                series=series,
                past_covariates=past_covs,
                future_covariates=future_covs,
                start=val_start,
                forecast_horizon=forecast_horizon,
                stride=stride,
                retrain=False,
                overlap_end=False,
                last_points_only=False,
                verbose=False,
                show_warnings=False,
            )
            wf_seconds = time.time() - t_wf_start

            rmses = []
            all_actual = []
            all_predicted = []
            for fcast in backtest_forecasts:
                if fcast.end_time() >= val_end:
                    continue
                actual = series.slice_intersect(fcast)
                act = actual.values().flatten()
                pred = fcast.values().flatten()
                err = act - pred
                rmses.append(float(np.sqrt(np.mean(err ** 2))))
                all_actual.append(act)
                all_predicted.append(pred)

                trial.report(float(np.mean(rmses)), len(rmses) - 1)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if not rmses:
                return float("inf")

            score = float(np.mean(rmses))

            val_metrics = compute_metrics(
                np.concatenate(all_actual), np.concatenate(all_predicted)
            )
            trial.set_user_attr("val_mae",   val_metrics["mae"])
            trial.set_user_attr("val_mape",  val_metrics["mape"])
            trial.set_user_attr("val_smape", val_metrics["smape"])
            trial.set_user_attr("val_r2",    val_metrics["r2"])

            residuos = pd.Series(
                np.concatenate(all_actual) - np.concatenate(all_predicted)
            ).dropna()
            try:
                resid_clean = residuos.to_numpy(dtype=float)
                resid_clean = resid_clean[~np.isnan(resid_clean)]
                acf_vals = acf(resid_clean, nlags=168)
                trial.set_user_attr("acf_24", float(abs(acf_vals[24])))
                trial.set_user_attr("acf_168", float(abs(acf_vals[168])))
            except Exception:
                trial.set_user_attr("acf_24", None)
                trial.set_user_attr("acf_168", None)
            trial.set_user_attr("residual_mean", float(residuos.mean()) if len(residuos) else None)

            trial.set_user_attr("hub_model_name",       hub_model_name)
            trial.set_user_attr("selected_past_cols",    selected_past_cols)
            trial.set_user_attr("selected_future_cols",  selected_future_cols)
            trial.set_user_attr("n_windows",             len(rmses))
            trial.set_user_attr("fit_seconds",           round(fit_seconds, 1))
            trial.set_user_attr("walkforward_seconds",   round(wf_seconds, 1))

            del model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

            return score

        except Exception as e:
            trial.set_user_attr("error", str(e))
            return float("inf")

    return objective

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    config = load_config(CONFIG_PATH)
    seed   = config.get("project", {}).get("random_seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    LOG_DIR    = Path(config.get("output", {}).get("logs", "src/optimization/logs"))
    OUTPUT_DIR = Path(config.get("output", {}).get("optimization", "output/optimization"))
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("chronos2_optuna")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_DIR / "chronos2_optimization.log")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")
    logger.info(f"Dispositivo: {device}")

    dataset_cfg = config.get("dataset", {})
    csv_path    = dataset_cfg.get("input_file_chronos2", r"data\processed\demandas_horarias_2023_2026[chronos2].csv")
    target_col  = dataset_cfg.get("target", "TOTAL_SIP")
    period_cfg  = dataset_cfg.get("period", {})

    print(f"Cargando datos desde {csv_path}...")
    df = pd.read_csv(csv_path)
    

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()

    period_start = pd.to_datetime(str(period_cfg.get("start", "2023-01-01")))
    period_end   = pd.to_datetime(str(period_cfg.get("end", "2026-01-01")))
    period_hours = (period_end - period_start).total_seconds() / 3600

    split_cfg   = config.get("split", {})
    train_ratio = split_cfg.get("train_ratio", 0.70)
    val_ratio   = split_cfg.get("validation_ratio", 0.15)

    train_hours = int(period_hours * train_ratio)
    val_hours   = int(period_hours * val_ratio)

    VAL_START  = period_start + pd.Timedelta(hours=train_hours)
    VAL_END    = period_start + pd.Timedelta(hours=train_hours + val_hours)
    TEST_START = VAL_END

    df = df.loc[(df.index >= period_start) & (df.index <= period_end)]

    missing = [c for c in [target_col] + ALL_PAST_COLS + ALL_FUTURE_COLS if c not in df.columns]
    if missing:
        logger.warning(f"Columnas faltantes (se ignorarán en la búsqueda): {missing}")
        print(f"Advertencia: columnas faltantes: {missing}")

    bt_cfg           = config.get("backtesting", {})
    FORECAST_HORIZON = bt_cfg.get("forecast_horizon", 24)
    STRIDE           = bt_cfg.get("stride", 24)
    CONTEXT_WINDOW   = bt_cfg.get("context_window", 168)

    n_val     = len(df[(df.index >= VAL_START) & (df.index < VAL_END)])
    n_windows = (n_val - FORECAST_HORIZON) // STRIDE + 1 if n_val > FORECAST_HORIZON else 0

    print(f"Val:   ({VAL_START} -> {VAL_END})")
    print(f"Test:  (desde {TEST_START}, reservado)")
    print(f"Walk-forward: horizon={FORECAST_HORIZON}h, stride={STRIDE}h, context_window={CONTEXT_WINDOW}h (~{n_windows} ventanas)")
    print(f"Variantes candidatas: {HUB_MODEL_VARIANTS}")
    print(f"Past covariates candidatas: {ALL_PAST_COLS}")
    print(f"Future covariates candidatas: {ALL_FUTURE_COLS}")
    print()

    logger.info(f"Split: VAL_START={VAL_START}, VAL_END={VAL_END}, TEST_START={TEST_START}")
    logger.info(f"Backtesting: horizon={FORECAST_HORIZON}, stride={STRIDE}, context_window={CONTEXT_WINDOW}")

    series_ts      = TimeSeries.from_dataframe(df[[target_col]], freq="h")
    past_covs_ts   = TimeSeries.from_dataframe(df[ALL_PAST_COLS], freq="h") if ALL_PAST_COLS else None
    future_covs_ts = TimeSeries.from_dataframe(df[ALL_FUTURE_COLS], freq="h") if ALL_FUTURE_COLS else None

    opt_cfg    = config.get("optimization", {})
    N_TRIALS   = opt_cfg.get("trials_chronos2", 30)
    STUDY_NAME = opt_cfg.get("study_name", "forecasting") + "_chronos2"
    DIRECTION  = opt_cfg.get("direction", "minimize")
    TIMEOUT    = opt_cfg.get("timeout_seconds", None)
    N_JOBS     = 1 if device == "cuda" else opt_cfg.get("n_jobs", 1)

    sampler_type = opt_cfg.get("sampler", "TPESampler")
    if sampler_type == "TPESampler":
        sampler = optuna.samplers.TPESampler(seed=seed)
    elif sampler_type == "RandomSampler":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    elif sampler_type == "CmaEsSampler":
        sampler = optuna.samplers.CmaEsSampler(seed=seed)
    else:
        logger.warning(f"Sampler '{sampler_type}' no reconocido, usando TPESampler")
        sampler = optuna.samplers.TPESampler(seed=seed)

    pruner_type = opt_cfg.get("pruner", "MedianPruner")

    if pruner_type == "MedianPruner":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=3, n_warmup_steps=2, interval_steps=1, n_min_trials=3
        )
    elif pruner_type == "SuccessiveHalvingPruner":
        pruner = optuna.pruners.SuccessiveHalvingPruner()
    elif pruner_type in (None, "NopPruner", "none", "None"):
        pruner = optuna.pruners.NopPruner()
    else:
        logger.warning(f"Pruner '{pruner_type}' no reconocido, usando MedianPruner")
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=3, n_warmup_steps=2, interval_steps=1, n_min_trials=3
        )

    chronos_cfg        = config.get("chronos2", {})
    chronos_search_cfg = chronos_cfg.get("search", {})

    objective_fn = build_objective(
        series=series_ts,
        past_covs_full=past_covs_ts,
        future_covs_full=future_covs_ts,
        val_start=VAL_START,
        val_end=VAL_END,
        forecast_horizon=FORECAST_HORIZON,
        stride=STRIDE,
        context_window=CONTEXT_WINDOW,
        chronos_search_cfg=chronos_search_cfg,
        device=device,
    )

    def progress_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        value = "inf" if trial.value == float("inf") else f"{trial.value:.4f}"
        fit_s = trial.user_attrs.get("fit_seconds", "?")
        wf_s  = trial.user_attrs.get("walkforward_seconds", "?")
        n_win = trial.user_attrs.get("n_windows", "?")
        state = trial.state.name
        best  = "inf" if study.best_value == float("inf") else f"{study.best_value:.4f}"
        acf24 = trial.user_attrs.get("acf_24")
        acf168 = trial.user_attrs.get("acf_168")
        acf24_s = "?" if acf24 is None else f"{acf24:.4f}"
        acf168_s = "?" if acf168 is None else f"{acf168:.4f}"
        print(f"[trial {trial.number:>3d}] RMSE={value:>10} | acf24={acf24_s} | acf168={acf168_s} | fit={fit_s:>6}s | wf={wf_s:>6}s ({n_win} win) | {state:8s} | best={best}")

    print(f"Iniciando optimización Chronos-2: {N_TRIALS} trials, n_jobs={N_JOBS}")
    print("=" * 80)

    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction=DIRECTION,
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(
        objective_fn,
        n_trials=N_TRIALS,
        timeout=TIMEOUT,
        n_jobs=N_JOBS,
        callbacks=[progress_callback],
    )

    logger.info(f"Optimización Chronos-2 completada. Best RMSE (val): {study.best_value:.4f}")

    print("\nCalculando métricas de test para cada trial completado...")
    trials_data = []
    for trial in study.trials:
        row = {
            "trial_number":        trial.number,
            "rmse_val":            trial.value if trial.value != float("inf") else "inf",
            "state":               trial.state.name,
            "val_mae":             trial.user_attrs.get("val_mae"),
            "val_mape":            trial.user_attrs.get("val_mape"),
            "val_smape":           trial.user_attrs.get("val_smape"),
            "val_r2":              trial.user_attrs.get("val_r2"),
            "hub_model_name":      trial.user_attrs.get("hub_model_name"),
            "selected_past_cols":  str(trial.user_attrs.get("selected_past_cols", [])),
            "selected_future_cols": str(trial.user_attrs.get("selected_future_cols", [])),
            "fit_seconds":         trial.user_attrs.get("fit_seconds"),
            "walkforward_seconds": trial.user_attrs.get("walkforward_seconds"),
            "n_windows":           trial.user_attrs.get("n_windows"),
            "acf_24":              trial.user_attrs.get("acf_24"),
            "acf_168":             trial.user_attrs.get("acf_168"),
            "residual_mean":       trial.user_attrs.get("residual_mean"),
        }

        if trial.value is not None and trial.value != float("inf"):
            try:
                hub_name     = trial.user_attrs.get("hub_model_name")
                past_cols    = trial.user_attrs.get("selected_past_cols", [])
                future_cols  = trial.user_attrs.get("selected_future_cols", [])

                test_model = Chronos2Model(
                    input_chunk_length=CONTEXT_WINDOW,
                    output_chunk_length=FORECAST_HORIZON,
                    hub_model_name=hub_name,
                    pl_trainer_kwargs={
                        "accelerator": "gpu" if device == "cuda" else "cpu",
                        "devices": [0] if device == "cuda" else "auto",
                        "enable_progress_bar": False,
                        "enable_model_summary": False,
                        "logger": False,
                    },
                )

                trial_past_covs   = past_covs_ts[past_cols] if past_cols and past_covs_ts is not None else None
                trial_future_covs = future_covs_ts[future_cols] if future_cols and future_covs_ts is not None else None

                trainval_series      = series_ts.drop_after(TEST_START)
                trainval_past_covs   = trial_past_covs.drop_after(TEST_START) if trial_past_covs is not None else None
                trainval_future_covs = trial_future_covs.drop_after(TEST_START) if trial_future_covs is not None else None

                test_model.fit(
                    series=trainval_series,
                    past_covariates=trainval_past_covs,
                    future_covariates=trainval_future_covs,
                    verbose=False,
                )

                backtest_test = test_model.historical_forecasts(
                    series=series_ts,
                    past_covariates=trial_past_covs,
                    future_covariates=trial_future_covs,
                    start=TEST_START,
                    forecast_horizon=FORECAST_HORIZON,
                    stride=STRIDE,
                    retrain=False,
                    overlap_end=False,
                    last_points_only=False,
                    verbose=False,
                    show_warnings=False,
                )

                all_actual, all_predicted = [], []
                for fcast in backtest_test:
                    actual = series_ts.slice_intersect(fcast)
                    all_actual.append(actual.values().flatten())
                    all_predicted.append(fcast.values().flatten())

                if all_actual:
                    y_true = np.concatenate(all_actual)
                    y_pred = np.concatenate(all_predicted)
                    tm = compute_metrics(y_true, y_pred)
                    row["rmse_test"]  = round(tm["rmse"], 4)
                    row["mae_test"]   = round(tm["mae"], 4)
                    row["mape_test"]  = round(tm["mape"], 4)
                    row["smape_test"] = round(tm["smape"], 4)
                    row["r2_test"]    = round(tm["r2"], 4)
                else:
                    row["rmse_test"] = row["mae_test"] = row["mape_test"] = row["smape_test"] = row["r2_test"] = None

                del test_model
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                row["rmse_test"] = row["mae_test"] = row["mape_test"] = row["smape_test"] = row["r2_test"] = None
                row["test_error"] = str(e)
        else:
            row["rmse_test"] = row["mae_test"] = row["mape_test"] = row["smape_test"] = row["r2_test"] = None

        trials_data.append(row)
        print(f"  trial {trial.number}: rmse_val={row['rmse_val']}, rmse_test={row.get('rmse_test')}")

    trials_df  = pd.DataFrame(trials_data)
    csv_output = OUTPUT_DIR / "chronos2_trials.csv"
    trials_df.to_csv(csv_output, index=False)
    print(f"\n  Trials exportados a: {csv_output}")
    logger.info(f"Trials exportados a: {csv_output}")

    rmse_val_num = pd.to_numeric(trials_df["rmse_val"], errors="coerce")
    rmse_val_num = rmse_val_num.where(np.isfinite(rmse_val_num))
    if rmse_val_num.notna().sum() == 0:
        print("No se encontró ningún trial exitoso con validación válida.")
        logger.error("No se encontró ningún trial exitoso")
        sys.exit(1)

    best_row = trials_df.loc[rmse_val_num.idxmin()]

    print("\n" + "=" * 80)
    print("MEJOR TRIAL SEGÚN RMSE_VAL (validación)")
    print("=" * 80)
    print(f"  trial_number   = {best_row['trial_number']}")
    print(f"  hub_model_name = {best_row['hub_model_name']}")
    print(f"  past_cols      = {best_row['selected_past_cols']}")
    print(f"  future_cols    = {best_row['selected_future_cols']}")
    print("\n  --- Validación (sin sufijo) ---")
    print(f"  rmse           = {best_row['rmse_val']}")
    print(f"  mae            = {best_row.get('val_mae')}")
    print(f"  mape           = {best_row.get('val_mape')}")
    print(f"  smape          = {best_row.get('val_smape')}")
    print(f"  r2             = {best_row.get('val_r2')}")
    print("\n  --- Test (holdout) ---")
    print(f"  rmse_test      = {best_row['rmse_test']}")
    print(f"  mae_test       = {best_row['mae_test']}")
    print(f"  mape_test      = {best_row['mape_test']}")
    print(f"  smape_test     = {best_row['smape_test']}")
    print(f"  r2_test        = {best_row['r2_test']}")

    if "chronos2" not in config:
        config["chronos2"] = {}

    config["chronos2"]["best"] = {}

    bs = config["chronos2"]["best"]
    bs["hub_model_name"]      = str(best_row["hub_model_name"])
    bs["past_covariates"]     = eval(best_row["selected_past_cols"]) if best_row["selected_past_cols"] != "[]" else []
    bs["future_covariates"]   = eval(best_row["selected_future_cols"]) if best_row["selected_future_cols"] != "[]" else []
    bs["input_chunk_length"]  = int(CONTEXT_WINDOW)
    bs["output_chunk_length"] = int(FORECAST_HORIZON)

    def _as_float(value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if np.isfinite(number) else None

    bs["rmse"]                = _as_float(best_row["rmse_val"])
    bs["mae"]                 = _as_float(best_row["val_mae"])
    bs["mape"]                = _as_float(best_row["val_mape"])
    bs["smape"]               = _as_float(best_row["val_smape"])
    bs["r2"]                  = _as_float(best_row["val_r2"])
    bs["rmse_test"]           = _as_float(best_row["rmse_test"])
    bs["mae_test"]            = _as_float(best_row["mae_test"])
    bs["mape_test"]           = _as_float(best_row["mape_test"])
    bs["smape_test"]          = _as_float(best_row["smape_test"])
    bs["r2_test"]             = _as_float(best_row["r2_test"])
    bs["trial_number"]        = int(best_row["trial_number"])
    bs["datetime"]            = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    save_config(config, CONFIG_PATH)
    print(f"\n  Mejor configuración guardada en 'chronos2.best' en {CONFIG_PATH}")
    logger.info(f"Guardado en chronos2.best: {CONFIG_PATH}")

    n_failed    = int((trials_df["rmse_val"] == "inf").sum())
    n_completed = len(trials_df) - n_failed
    print(f"\n--- Statistics ---")
    print(f"  Completed:    {n_completed}/{len(trials_df)}")
    print(f"  Failed:       {n_failed}")

    print("\n" + "=" * 80)