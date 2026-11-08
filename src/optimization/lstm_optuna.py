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
import optuna
import torch
torch.backends.fp32_precision = "ieee"
from statsmodels.tsa.stattools import acf
from ruamel.yaml import YAML
from darts import TimeSeries
from darts.models import RNNModel
from darts.dataprocessing.transformers import Scaler

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

torch.set_num_threads(1)

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

def build_objective(
    series_scaled: TimeSeries,
    future_covs_scaled: TimeSeries,
    scaler_target: Scaler,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
    test_start: pd.Timestamp,
    forecast_horizon: int,
    stride: int,
    context_window: int,
    lstm_search_cfg: dict,
    device: str,
    random_seed: int,
    col_map: dict,
):

    resolved_candidates = {
        "use_temperatura": [col_map["temp"]],
        "use_humedad":     [col_map["hum"]],
        "use_temp_lags":   ["temp_lag_24h"],
        "use_hum_lags":    ["hum_lag_24h"],
        "use_feriado":     [col_map["feriado"]] if col_map["feriado"] else [],
        "use_dia_anio":    ["dia_anio_sin", "dia_anio_cos"],
        "use_target_lags": ["target_lag_24"],
    }

    def _to_num(x):

        if isinstance(x, (int, float)):
            return x
        s = str(x)
        try:
            v = float(s)
        except Exception:
            raise ValueError(f"Cannot convert bound value '{x}' to number")
        if v.is_integer():
            return int(v)
        return v

    def get_bounds(name, default):
        b = lstm_search_cfg.get(name, default)

        if isinstance(b, (list, tuple)):
            if len(b) != 2:
                raise ValueError(f"Bounds for {name} must be length 2, got: {b}")
            return [_to_num(b[0]), _to_num(b[1])]

        return [_to_num(b), _to_num(b)]

    hidden_bounds = get_bounds("hidden_size",   [32, 128])
    layer_bounds  = get_bounds("num_layers",    [1, 2])
    drop_bounds   = get_bounds("dropout",       [0.0, 0.3])
    batch_bounds  = get_bounds("batch_size",    [16, 64])
    lr_bounds     = get_bounds("learning_rate", [1e-4, 1e-2])
    epoch_bounds  = get_bounds("epochs",        [5, 20])

    def objective(trial: optuna.Trial) -> float:

        hidden_size = trial.suggest_int(  "hidden_size",   hidden_bounds[0], hidden_bounds[1], step=16)
        num_layers  = trial.suggest_int(  "num_layers",    layer_bounds[0],  layer_bounds[1])
        dropout     = trial.suggest_float("dropout",       drop_bounds[0],   drop_bounds[1])
        batch_size  = trial.suggest_int(  "batch_size",    batch_bounds[0],  batch_bounds[1], step=16)
        lr          = trial.suggest_float("learning_rate", lr_bounds[0],     lr_bounds[1], log=True)
        epochs      = trial.suggest_int(  "epochs",        epoch_bounds[0],  epoch_bounds[1])

        selected_cols = list(FIXED_FUTURE_COLS)
        selected_candidate_cols = []

        for flag, cols in resolved_candidates.items():
            if not cols:
                continue
            if trial.suggest_categorical(flag, [True, False]):
                selected_cols.extend(cols)
                selected_candidate_cols.extend(cols)

        trial_future_covs = future_covs_scaled[selected_cols]

        try:
            t_fit_start = time.time()

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
                random_state=random_seed,
                pl_trainer_kwargs={
                    "accelerator": "gpu" if device == "cuda" else "cpu",
                    "devices": [0] if device == "cuda" else "auto",
                    "enable_progress_bar": True,
                    "enable_model_summary": False,
                    "logger": False,
                },
                force_reset=True,
                save_checkpoints=False,
            )

            train_series      = series_scaled.drop_after(val_start)
            train_future_covs = trial_future_covs.drop_after(val_start)

            model.fit(series=train_series, future_covariates=train_future_covs, verbose=False)
            fit_seconds = time.time() - t_fit_start

            t_wf_start = time.time()
            backtest_forecasts = model.historical_forecasts(
                series=series_scaled,
                future_covariates=trial_future_covs,
                start=val_start,
                forecast_horizon=forecast_horizon,
                stride=stride,
                retrain=False,
                overlap_end=False,
                last_points_only=False,
                verbose=False,
            )
            wf_seconds = time.time() - t_wf_start

            rmses = []
            all_actual = []
            all_predicted = []
            for fcast in backtest_forecasts:
                if fcast.end_time() >= val_end:
                    continue
                

                actual_scaled = series_scaled.slice_intersect(fcast)
                

                actual_real = scaler_target.inverse_transform(actual_scaled).values().flatten()
                pred_real   = scaler_target.inverse_transform(fcast).values().flatten()
                

                err = actual_real - pred_real
                rmses.append(float(np.sqrt(np.mean(err ** 2))))
                all_actual.append(actual_real)
                all_predicted.append(pred_real)

                trial.report(float(np.mean(rmses)), len(rmses) - 1)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if not rmses:
                return float("inf")

            score = float(np.mean(rmses))

            val_metrics = compute_metrics(
                np.concatenate(all_actual), np.concatenate(all_predicted)
            )

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

            try:
                num_params = sum(p.numel() for p in model.model.parameters())
                trial.set_user_attr("num_params", int(num_params))
            except Exception:
                trial.set_user_attr("num_params", None)

            test_metrics = None
            n_windows_test = 0
            try:
                backtest_test = model.historical_forecasts(
                    series=series_scaled,
                    future_covariates=trial_future_covs,
                    start=test_start,
                    forecast_horizon=forecast_horizon,
                    stride=stride,
                    retrain=False,
                    overlap_end=False,
                    last_points_only=False,
                    verbose=False,
                )

                test_actual = []
                test_predicted = []
                for fcast in backtest_test:
                    actual_scaled = series_scaled.slice_intersect(fcast)
                    actual_real = scaler_target.inverse_transform(actual_scaled).values().flatten()
                    pred_real   = scaler_target.inverse_transform(fcast).values().flatten()
                    test_actual.append(actual_real)
                    test_predicted.append(pred_real)

                if test_actual:
                    n_windows_test = len(test_actual)
                    test_metrics = compute_metrics(
                        np.concatenate(test_actual), np.concatenate(test_predicted)
                    )
            except Exception:
                test_metrics = None

            trial.set_user_attr("fixed_features",             FIXED_FUTURE_COLS)
            trial.set_user_attr("selected_candidate_features", selected_candidate_cols)
            trial.set_user_attr("all_features",                selected_cols)
            trial.set_user_attr("n_windows",                   len(rmses))
            trial.set_user_attr("fit_seconds",                 round(fit_seconds, 1))
            trial.set_user_attr("walkforward_seconds",         round(wf_seconds, 1))

            trial.set_user_attr("val_mae",   val_metrics["mae"])
            trial.set_user_attr("val_rmse",  val_metrics["rmse"])
            trial.set_user_attr("val_mape",  val_metrics["mape"])
            trial.set_user_attr("val_smape", val_metrics["smape"])
            trial.set_user_attr("val_r2",    val_metrics["r2"])

            if test_metrics:
                trial.set_user_attr("test_mae",   test_metrics["mae"])
                trial.set_user_attr("test_rmse",  test_metrics["rmse"])
                trial.set_user_attr("test_mape",  test_metrics["mape"])
                trial.set_user_attr("test_smape", test_metrics["smape"])
                trial.set_user_attr("test_r2",    test_metrics["r2"])
                trial.set_user_attr("n_windows_test", n_windows_test)

            del model
            gc.collect()

            return score

        except Exception as e:
            trial.set_user_attr("error", str(e))
            return float("inf")

    return objective

if __name__ == "__main__":
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.getLogger("darts").setLevel(logging.ERROR)

    config = load_config(CONFIG_PATH)
    seed   = config.get("project", {}).get("random_seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    LOG_DIR    = Path(config.get("output", {}).get("logs", "src/optimization/logs"))
    OUTPUT_DIR = Path(config.get("output", {}).get("optimization", "output/optimization"))
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("lstm_optuna")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_DIR / "lstm_optimization.log")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")
    logger.info(f"Dispositivo: {device}")

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
        try:
            start_ts = pd.to_datetime(str(period_cfg["start"]))
            end_ts = pd.to_datetime(str(period_cfg["end"]))
            logger.info(f"Aplicando periodo: {start_ts} -> {end_ts}")
            print(f"Aplicando periodo: {start_ts} -> {end_ts}")
            df = df.loc[start_ts:end_ts]
        except Exception as e:
            logger.error(f"Error al parsear period.start/end en config: {e}")
            print(f"Error al parsear period.start/end en config: {e}")
            sys.exit(1)

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
        logger.error(f"Columnas faltantes en el dataset: {missing}")
        print(f"Error: columnas faltantes: {missing}")
        sys.exit(1)

    split_cfg   = config.get("split", {})
    train_ratio = split_cfg.get("train_ratio", 0.70)
    val_ratio   = split_cfg.get("validation_ratio", 0.15)

    period_start = pd.to_datetime(str(period_cfg.get("start", "2023-01-01")))
    period_end   = pd.to_datetime(str(period_cfg.get("end", "2026-01-01")))
    period_hours = (period_end - period_start).total_seconds() / 3600

    train_hours = int(period_hours * train_ratio)
    val_hours   = int(period_hours * val_ratio)

    VAL_START  = period_start + pd.Timedelta(hours=train_hours)
    VAL_END    = period_start + pd.Timedelta(hours=train_hours + val_hours)
    TEST_START = VAL_END

    n_total        = len(df)
    val_start_idx  = int(df.index.searchsorted(VAL_START))
    test_start_idx = int(df.index.searchsorted(TEST_START))

    bt_cfg           = config.get("backtesting", {})
    FORECAST_HORIZON = bt_cfg.get("forecast_horizon", 24)
    STRIDE           = bt_cfg.get("stride", 24)
    CONTEXT_WINDOW   = bt_cfg.get("context_window", 24 * 7)

    n_train   = val_start_idx
    n_val     = test_start_idx - val_start_idx
    n_test    = n_total - test_start_idx
    n_windows = (n_val - FORECAST_HORIZON) // STRIDE + 1

    print(f"Train: {n_train} obs (hasta {VAL_START})")
    print(f"Val:   {n_val} obs ({VAL_START} -> {VAL_END})")
    print(f"Test:  {n_test} obs (desde {TEST_START}, reservado)")
    print(f"Walk-forward: horizon={FORECAST_HORIZON}h, stride={STRIDE}h, context_window={CONTEXT_WINDOW}h (~{n_windows} ventanas)")
    print(f"input_chunk_length = context_window = {CONTEXT_WINDOW}h (fijado para comparabilidad con SARIMAX)")
    print(f"Fixed future covariates:     {FIXED_FUTURE_COLS}")
    print(f"Candidate groups:            {list(CANDIDATE_GROUPS.keys())}")
    print()

    logger.info(
        f"Split: VAL_START={VAL_START}, VAL_END={VAL_END}, TEST_START={TEST_START} | "
        f"Backtesting: horizon={FORECAST_HORIZON}, stride={STRIDE}, context_window={CONTEXT_WINDOW}"
    )

    all_future_cols = (
        FIXED_FUTURE_COLS
        + [col_map["temp"], col_map["hum"]]
        + ["dia_anio_sin", "dia_anio_cos"]
        + ["temp_lag_24h", "hum_lag_24h", "target_lag_24"]
        + ([col_map["feriado"]] if col_map["feriado"] else [])
    )

    seen = set()
    all_future_cols = [c for c in all_future_cols if not (c in seen or seen.add(c))]

    series_ts      = TimeSeries.from_dataframe(df[[target_col]], freq="h")
    future_covs_ts = TimeSeries.from_dataframe(df[all_future_cols], freq="h")

    scaler_target = Scaler()
    scaler_target.fit(series_ts.drop_after(VAL_START))
    series_scaled = scaler_target.transform(series_ts)

    scaler_future = Scaler()
    scaler_future.fit(future_covs_ts.drop_after(VAL_START))
    future_covs_scaled = scaler_future.transform(future_covs_ts)

    opt_cfg    = config.get("optimization", {})
    N_TRIALS   = opt_cfg.get("trials_lstm", 50)
    STUDY_NAME = opt_cfg.get("study_name", "forecasting") + "_lstm"
    DIRECTION  = opt_cfg.get("direction", "minimize")
    TIMEOUT    = opt_cfg.get("timeout_seconds", None)
    N_JOBS     = opt_cfg.get("n_jobs", 1)

    sampler_type = opt_cfg.get("sampler", "TPESampler")
    if sampler_type == "TPESampler":
        sampler = optuna.samplers.TPESampler(seed=seed)
    elif sampler_type == "RandomSampler":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    elif sampler_type == "CmaEsSampler":
        sampler = optuna.samplers.CmaEsSampler(seed=seed)
    else:
        msg = f"Sampler '{sampler_type}' no reconocido, usando TPESampler"
        logger.warning(msg)
        print(f"  {msg}")
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
        msg = f"Pruner '{pruner_type}' no reconocido, usando MedianPruner"
        logger.warning(msg)
        print(f"  {msg}")
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=3, n_warmup_steps=2, interval_steps=1, n_min_trials=3
        )

    lstm_cfg        = config.get("lstm", {})
    lstm_search_cfg = lstm_cfg.get("search", {})

    objective_fn = build_objective(
        series_scaled=series_scaled,
        future_covs_scaled=future_covs_scaled,
        scaler_target=scaler_target,
        val_start=VAL_START,
        val_end=VAL_END,
        test_start=TEST_START,
        forecast_horizon=FORECAST_HORIZON,
        stride=STRIDE,
        context_window=CONTEXT_WINDOW,
        lstm_search_cfg=lstm_search_cfg,
        device=device,
        random_seed=seed,
        col_map=col_map,
    )

    def progress_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        value = "inf" if trial.value == float("inf") else f"{trial.value:.4f}"
        fit_s = trial.user_attrs.get("fit_seconds", "?")
        wf_s  = trial.user_attrs.get("walkforward_seconds", "?")
        n_win = trial.user_attrs.get("n_windows", "?")
        state = trial.state.name
        best  = "inf" if study.best_value == float("inf") else f"{study.best_value:.4f}"
        val_mae = trial.user_attrs.get("val_mae")
        test_rmse = trial.user_attrs.get("test_rmse")
        val_mae_s = "?" if val_mae is None else f"{val_mae:>10.4f}"
        test_rmse_s = "?" if test_rmse is None else f"{test_rmse:>10.4f}"
        acf24 = trial.user_attrs.get("acf_24")
        acf168 = trial.user_attrs.get("acf_168")
        acf24_s = "?" if acf24 is None else f"{acf24:.4f}"
        acf168_s = "?" if acf168 is None else f"{acf168:.4f}"
        print(f"[trial {trial.number:>3d}] RMSE={value:>10} | val_mae={val_mae_s} | test_rmse={test_rmse_s} | acf24={acf24_s} | acf168={acf168_s} | fit={fit_s:>6}s | wf={wf_s:>6}s ({n_win} win) | {state:8s} | best={best}")

    print(f"Iniciando optimización LSTM: {N_TRIALS} trials, n_jobs={N_JOBS}")
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

    logger.info(f"Optimización LSTM completada. Best RMSE (val): {study.best_value:.4f}")

    trials_data = []
    for trial in study.trials:
        row = {
            "trial_number":        trial.number,
            "value":               trial.value if trial.value != float("inf") else "inf",
            "state":               trial.state.name,
            "hidden_size":         trial.params.get("hidden_size"),
            "num_layers":          trial.params.get("num_layers"),
            "dropout":             trial.params.get("dropout"),
            "batch_size":          trial.params.get("batch_size"),
            "learning_rate":       trial.params.get("learning_rate"),
            "epochs":              trial.params.get("epochs"),
            "use_temperatura":     trial.params.get("use_temperatura"),
            "use_humedad":         trial.params.get("use_humedad"),
            "use_feriado":         trial.params.get("use_feriado"),
            "use_dia_anio":        trial.params.get("use_dia_anio"),
            "fit_seconds":         trial.user_attrs.get("fit_seconds"),
            "walkforward_seconds": trial.user_attrs.get("walkforward_seconds"),
            "n_windows":           trial.user_attrs.get("n_windows"),
            "selected_features":   str(trial.user_attrs.get("all_features", [])),
            "n_windows_test":      trial.user_attrs.get("n_windows_test"),
            "error":               trial.user_attrs.get("error"),
            "acf_24":              trial.user_attrs.get("acf_24"),
            "acf_168":             trial.user_attrs.get("acf_168"),
            "residual_mean":       trial.user_attrs.get("residual_mean"),
            "num_params":          trial.user_attrs.get("num_params"),
        }

        for m in ["mae", "rmse", "mape", "smape", "r2"]:
            row[f"val_{m}"]  = trial.user_attrs.get(f"val_{m}")
            row[f"test_{m}"] = trial.user_attrs.get(f"test_{m}")
        trials_data.append(row)

    trials_df  = pd.DataFrame(trials_data)
    csv_output = OUTPUT_DIR / "lstm_trials.csv"
    trials_df.to_csv(csv_output, index=False)
    print(f"\n  Trials exportados a: {csv_output}")
    logger.info(f"Trials exportados a: {csv_output}")

    print("\n" + "=" * 80)
    print("RESULTADOS DE LA BÚSQUEDA (sobre validación)")
    print("=" * 80)

    best = study.best_trial
    if best.value == float("inf"):
        print("No se encontró ningún trial exitoso.")
        logger.error("No se encontró ningún trial exitoso")
        sys.exit(1)

    best_hidden   = best.params.get("hidden_size")
    best_layers   = best.params.get("num_layers")
    best_dropout  = best.params.get("dropout")
    best_batch    = best.params.get("batch_size")
    best_lr       = best.params.get("learning_rate")
    best_epochs   = best.params.get("epochs")
    best_features = best.user_attrs.get("all_features", FIXED_FUTURE_COLS)
    best_cand     = best.user_attrs.get("selected_candidate_features", [])

    print(f"\nBest RMSE (val): {study.best_value:.4f} ({best.user_attrs.get('n_windows', '?')} ventanas)")
    print(f"  hidden_size              = {best_hidden}")
    print(f"  num_layers               = {best_layers}")
    print(f"  dropout                  = {best_dropout:.4f}")
    print(f"  batch_size               = {best_batch}")
    print(f"  learning_rate            = {best_lr:.6f}")
    print(f"  epochs                   = {best_epochs}")
    print(f"  future_covariates fixed:    {FIXED_FUTURE_COLS}")
    print(f"  future_covariates selected: {best_cand}")

    print(f"\n--- Métricas de validación (mejor trial) ---")
    print(f"  mae   = {best.user_attrs.get('val_mae',   float('nan')):.4f}")
    print(f"  rmse  = {study.best_value:.4f}")
    print(f"  mape  = {best.user_attrs.get('val_mape',  float('nan')):.4f}")
    print(f"  smape = {best.user_attrs.get('val_smape', float('nan')):.4f}")
    print(f"  r2    = {best.user_attrs.get('val_r2',    float('nan')):.4f}")

    if best.user_attrs.get("test_rmse") is not None:
        print(f"\n--- Métricas de test por trial (walk-forward, retrain=False) ---")
        print(f"  mae   = {best.user_attrs.get('test_mae',   float('nan')):.4f}")
        print(f"  rmse  = {best.user_attrs.get('test_rmse',  float('nan')):.4f}")
        print(f"  mape  = {best.user_attrs.get('test_mape',  float('nan')):.4f}")
        print(f"  smape = {best.user_attrs.get('test_smape', float('nan')):.4f}")
        print(f"  r2    = {best.user_attrs.get('test_r2',    float('nan')):.4f}")
        print(f"  ({best.user_attrs.get('n_windows_test', '?')} ventanas)")

    print("\n" + "=" * 80)
    print("EVALUACIÓN FINAL SOBRE TEST (holdout nunca visto durante la búsqueda)")
    print("=" * 80)

    test_metrics = None
    try:
        best_future_covs = future_covs_scaled[best_features]
        trainval_series  = series_scaled.drop_after(TEST_START)
        trainval_future  = best_future_covs.drop_after(TEST_START)

        final_model = RNNModel(
            model="LSTM",
            input_chunk_length=CONTEXT_WINDOW,
            output_chunk_length=1,
            training_length=CONTEXT_WINDOW + 1,
            hidden_dim=best_hidden,
            n_rnn_layers=best_layers,
            dropout=best_dropout,
            batch_size=best_batch,
            optimizer_kwargs={"lr": best_lr},
            n_epochs=best_epochs,
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

        print("Reentrenando mejor modelo con train + val...")
        final_model.fit(series=trainval_series, future_covariates=trainval_future, verbose=False)

        backtest_test = final_model.historical_forecasts(
            series=series_scaled,
            future_covariates=best_future_covs,
            start=TEST_START,
            forecast_horizon=FORECAST_HORIZON,
            stride=STRIDE,
            retrain=False,
            overlap_end=False,
            last_points_only=False,
            verbose=False,
        )

        all_actual    = []
        all_predicted = []
        for fcast in backtest_test:
            actual_scaled = series_scaled.slice_intersect(fcast)

            actual_real = scaler_target.inverse_transform(actual_scaled).values().flatten()
            pred_real   = scaler_target.inverse_transform(fcast).values().flatten()
            all_actual.append(actual_real)
            all_predicted.append(pred_real)

        if all_actual:
            y_true = np.concatenate(all_actual)
            y_pred = np.concatenate(all_predicted)
            test_metrics   = compute_metrics(y_true, y_pred)
            n_windows_test = len(all_actual)
            print(f"\n  Métricas en test ({n_windows_test} ventanas):")
            for k, v in test_metrics.items():
                print(f"    {k:6s} = {v:.4f}")
            logger.info(f"Métricas test LSTM: {test_metrics}")
        else:
            print("  No se generaron ventanas de test.")
            logger.warning("No se generaron ventanas de test")

    except Exception as e:
        print(f"  Error evaluando sobre test: {e}")
        logger.error(f"Error evaluando sobre test LSTM: {e}")

    if "lstm" not in config:
        config["lstm"] = {}

    config["lstm"]["best"] = {}

    bs = config["lstm"]["best"]
    bs["hidden_size"]         = int(best_hidden)
    bs["num_layers"]          = int(best_layers)
    bs["dropout"]             = float(round(best_dropout, 4))
    bs["batch_size"]          = int(best_batch)
    bs["learning_rate"]       = float(best_lr)
    bs["epochs"]              = int(best_epochs)
    bs["input_chunk_length"]  = int(CONTEXT_WINDOW)
    bs["output_chunk_length"] = int(FORECAST_HORIZON)
    bs["fixed_features"]      = FIXED_FUTURE_COLS
    bs["selected_features"]   = best_cand

    def _r4(value):
        if value is None:
            return None
        number = float(value)
        return round(number, 4) if np.isfinite(number) else None

    bs["rmse"]                = _r4(best.value)
    bs["mae"]                 = _r4(best.user_attrs.get("val_mae"))
    bs["mape"]                = _r4(best.user_attrs.get("val_mape"))
    bs["smape"]               = _r4(best.user_attrs.get("val_smape"))
    bs["r2"]                  = _r4(best.user_attrs.get("val_r2"))
    bs["rmse_test"]           = _r4(test_metrics["rmse"])  if test_metrics else None
    bs["mae_test"]            = _r4(test_metrics["mae"])   if test_metrics else None
    bs["mape_test"]           = _r4(test_metrics["mape"])  if test_metrics else None
    bs["smape_test"]          = _r4(test_metrics["smape"]) if test_metrics else None
    bs["r2_test"]             = _r4(test_metrics["r2"])    if test_metrics else None
    bs["trial_number"]        = int(best.number)
    bs["datetime"]            = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    save_config(config, CONFIG_PATH)
    print(f"\n  Mejor configuración guardada en 'lstm.best' dentro de {CONFIG_PATH} (comentarios preservados)")
    logger.info(f"Guardado en lstm.best: {CONFIG_PATH}")

    n_failed    = sum(1 for t in study.trials if t.value == float("inf"))
    n_pruned    = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    n_completed = len(study.trials) - n_failed - n_pruned

    print(f"\n--- Statistics ---")
    print(f"  Completed:    {n_completed}/{len(study.trials)}")
    print(f"  Failed:       {n_failed}")
    print(f"  Pruned:       {n_pruned}")
    print(f"  Success rate: {100 * n_completed / len(study.trials):.1f}%")

    try:
        importances = optuna.importance.get_param_importances(study)
        print(f"\n--- Parameter Importance ---")
        for k, v in sorted(importances.items(), key=lambda x: -x[1]):
            print(f"  {k:25s} {v:.3f}")
        logger.info(f"Parameter importance: {dict(sorted(importances.items(), key=lambda x: -x[1]))}")
    except Exception as e:
        logger.warning(f"No se pudo calcular parameter importance: {e}")

    print(f"\n--- Top 10 Trials ---")
    completed_trials = [t for t in study.trials if t.value is not None and t.value != float("inf")]
    completed_trials.sort(key=lambda t: t.value)

    for i, t in enumerate(completed_trials[:10], 1):
        feats = t.user_attrs.get("all_features", [])
        print(f"\n  [{i}] RMSE={t.value:.4f}")
        print(f"      hidden={t.params.get('hidden_size')}  layers={t.params.get('num_layers')}  "
              f"dropout={t.params.get('dropout', 0):.3f}  batch={t.params.get('batch_size')}  "
              f"lr={t.params.get('learning_rate', 0):.6f}  epochs={t.params.get('epochs')}")
        print(f"      features: {', '.join(feats)}")

    logger.info(
        f"Best trial LSTM #{best.number}: RMSE(val)={study.best_value:.4f}, "
        f"features={best_features}, test_metrics={test_metrics}"
    )

    print("\n" + "=" * 80)