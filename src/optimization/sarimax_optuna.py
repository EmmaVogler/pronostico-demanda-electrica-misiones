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
import yaml
import pandas as pd
import numpy as np
import optuna
from statsmodels.tsa.stattools import acf
from statsforecast.models import ARIMA
import statsforecast.arima as _sf_arima

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

CONFIG_PATH = Path("config.yaml")

def load_config(config_path: Path = CONFIG_PATH) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

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

    _INF_THRESHOLD = 300
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

def _sample_int_param(trial: optuna.Trial, name: str, bounds: list) -> int:
    if not bounds:
        raise ValueError(f"Las cotas de búsqueda para {name} no pueden estar vacias.")
    if len(bounds) == 1:
        return bounds[0]
    return trial.suggest_int(name, bounds[0], bounds[1])

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

def build_objective(
    target: pd.Series,
    covariates: pd.DataFrame,

    val_start_idx: int,
    test_start_idx: int,
    forecast_horizon: int,
    stride: int,
    context_window: int,

    maxiter: int,
    seasonal_period: int,
    seasonal_period_options: list,
    method: str,
    initialization: str,
    enforce_stationarity: bool,
    enforce_invertibility: bool,
    search_cfg: dict,
    fixed_features: list,
    candidates_cfg: dict,
    fourier_cfg: dict,
):
    p_bounds = search_cfg.get("p", [0, 3])
    d_bounds = search_cfg.get("d", [0])
    q_bounds = search_cfg.get("q", [0, 3])
    P_bounds = search_cfg.get("P", [0, 2])
    D_bounds = search_cfg.get("D", [0, 1])
    Q_bounds = search_cfg.get("Q", [0, 2])

    fourier_periods = fourier_cfg.get("periods", {"daily": 24.0, "weekly": 168.0, "yearly": 8765.82})

    k_daily_bounds = fourier_cfg.get("k_daily", [1, 6])
    k_weekly_bounds = fourier_cfg.get("k_weekly", [1, 4])
    k_yearly_bounds = fourier_cfg.get("k_yearly", [1, 3])

    def objective(trial: optuna.Trial) -> float:

        p = _sample_int_param(trial, "p", p_bounds)
        d = _sample_int_param(trial, "d", d_bounds)
        q = _sample_int_param(trial, "q", q_bounds)
        P = _sample_int_param(trial, "P", P_bounds)
        D = _sample_int_param(trial, "D", D_bounds)
        Q = _sample_int_param(trial, "Q", Q_bounds)

        s = trial.suggest_categorical("seasonal_period", seasonal_period_options)

        k_daily = _sample_int_param(trial, "k_daily", k_daily_bounds)
        k_weekly = _sample_int_param(trial, "k_weekly", k_weekly_bounds)
        k_yearly = _sample_int_param(trial, "k_yearly", k_yearly_bounds)

        selected_features = list(fixed_features)
        selected_candidate_features = []

        for group_name, group_info in candidates_cfg.items():
            flag_name = group_info.get("flag", f"use_{group_name}")
            feats = group_info.get("features", [])
            if trial.suggest_categorical(flag_name, [True, False]):
                selected_features.extend(feats)
                selected_candidate_features.extend(feats)

        fourier_df = _build_fourier_df(target.index, fourier_periods, k_daily, k_weekly, k_yearly)
        fourier_features = list(fourier_df.columns)
        selected_features.extend(fourier_features)
        selected_candidate_features.extend(fourier_features)

        cov_features = [f for f in selected_features if f in covariates.columns]

        y_all = target.to_numpy(dtype=float)
        X_all = np.hstack(
            [covariates[cov_features].to_numpy(dtype=float), fourier_df.to_numpy(dtype=float)]
        )

        train_idx = val_start_idx
        train_y = y_all[:train_idx]
        train_X = X_all[:train_idx]

        order = (p, d, q)
        seasonal_order = (P, D, Q)

        try:

            t_fit_start = time.time()
            model = ARIMA(order=order, season_length=s, seasonal_order=seasonal_order, method=method)
            model.fit(train_y, train_X)
            fit_seconds = time.time() - t_fit_start

            model_ = model.model_
            coefs = model_.get("coef", {})
            sigma2 = model_.get("sigma2", None)

            if not coefs or not all(np.isfinite(float(v)) for v in coefs.values()):
                return float("inf")
            if sigma2 is None or not np.isfinite(float(sigma2)) or float(sigma2) <= 0:
                return float("inf")
            fixed = {k: float(v) for k, v in coefs.items()}

            try:
                resid = np.asarray(model_.get("residuals"), dtype=float).ravel()
                resid_clean = resid[~np.isnan(resid)]
                acf_vals = acf(resid_clean, nlags=168)
                trial.set_user_attr("acf_24", float(abs(acf_vals[24])))
                trial.set_user_attr("acf_168", float(abs(acf_vals[168])))
            except Exception:
                trial.set_user_attr("acf_24", None)
                trial.set_user_attr("acf_168", None)

            loglik = model_.get("loglik")
            npar = len(coefs) + 1
            nobs = model_.get("nobs")
            aic = None
            bic = None
            if loglik is not None and np.isfinite(float(loglik)) and nobs:
                aic = -2.0 * float(loglik) + 2.0 * npar
                bic = -2.0 * float(loglik) + npar * float(np.log(nobs))
            trial.set_user_attr("aic", float(aic) if aic is not None and np.isfinite(float(aic)) else None)
            trial.set_user_attr("bic", float(bic) if bic is not None and np.isfinite(float(bic)) else None)

            del model
            gc.collect()

            val_target = target.iloc[val_start_idx:test_start_idx]
            n_val = len(val_target)
            n_windows = (n_val - forecast_horizon) // stride + 1
            if n_windows <= 0:
                return float("inf")

            start_idx = val_start_idx
            errors = []
            all_actuals = []
            all_predicted = []
            t_wf_start = time.time()

            for w in range(n_windows):
                abs_pos = start_idx + w * stride
                if abs_pos + forecast_horizon > len(target):
                    break

                act = y_all[abs_pos : abs_pos + forecast_horizon]
                pred = _fixed_window_forecast(
                    y_all, X_all, abs_pos, context_window, forecast_horizon,
                    order, seasonal_order, s, fixed, method,
                )
                window_rmse = np.sqrt(np.mean((act - pred) ** 2))
                errors.append(window_rmse)
                all_actuals.append(act)
                all_predicted.append(pred)

                trial.report(float(np.mean(errors)), w)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if not errors:
                return float("inf")

            score = float(np.mean(errors))

            val_metrics = compute_metrics(
                np.concatenate(all_actuals), np.concatenate(all_predicted)
            )

            test_metrics = None
            n_windows_test = 0
            try:
                test_target = target.iloc[test_start_idx:]
                n_test = len(test_target)
                n_windows_test = (n_test - forecast_horizon) // stride + 1
                if n_windows_test > 0:
                    test_actuals = []
                    test_predicted = []
                    for w in range(n_windows_test):
                        abs_pos = test_start_idx + w * stride
                        if abs_pos + forecast_horizon > len(target):
                            break

                        test_actuals.append(y_all[abs_pos : abs_pos + forecast_horizon])
                        test_predicted.append(
                            _fixed_window_forecast(
                                y_all, X_all, abs_pos, context_window, forecast_horizon,
                                order, seasonal_order, s, fixed, method,
                            )
                        )

                    if test_actuals:
                        test_metrics = compute_metrics(
                            np.concatenate(test_actuals), np.concatenate(test_predicted)
                        )
            except Exception:
                test_metrics = None

            trial.set_user_attr("fixed_features", fixed_features)
            trial.set_user_attr("selected_candidate_features", selected_candidate_features)
            trial.set_user_attr("features", selected_features)
            trial.set_user_attr("order", (p, d, q))
            trial.set_user_attr("seasonal_order", (P, D, Q, s))
            trial.set_user_attr("n_windows", len(errors))
            trial.set_user_attr("fit_seconds", round(fit_seconds, 1))
            trial.set_user_attr("walkforward_seconds", round(time.time() - t_wf_start, 1))

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

            return score

        except Exception as e:
            trial.set_user_attr("error", f"{type(e).__name__}: {e}")
            logging.getLogger(__name__).warning(
                "Trial %d falló (order=%s, seasonal_order=%s): %s: %s",
                trial.number, order, seasonal_order, type(e).__name__, e,
            )
            return float("inf")

    return objective

if __name__ == "__main__":

    config = load_config(CONFIG_PATH)
    storage_name = "sqlite:///sarimax_optimization.db"

    LOG_DIR = Path(config.get("output", {}).get("logs", "src/optimization/logs"))
    OUTPUT_DIR = Path(config.get("output", {}).get("optimization", "output/optimization"))
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(LOG_DIR / "sarimax_optimization.log")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

    dataset_cfg = config.get("dataset", {})
    csv_path = dataset_cfg.get("input_file_sarimax", "data/processed/demandas_horarias_2023_2026[sarimax].csv")
    date_col = dataset_cfg.get("datetime", {}).get("date_column", "FECHA")
    hour_col = dataset_cfg.get("datetime", {}).get("hour_column", "Horario")
    target_col = dataset_cfg.get("target", "TOTAL_SIP")
    period_cfg = dataset_cfg.get("period", {})

    logger.info(f"Cargando dataset desde {csv_path}...")
    print(f"Cargando dataset desde {csv_path}...")

    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(df[date_col]) + pd.to_timedelta(df[hour_col], unit="h")
    df = df.set_index("datetime").sort_index()

    if "start" in period_cfg and "end" in period_cfg and period_cfg["start"] and period_cfg["end"]:
        df = df.loc[period_cfg["start"] : period_cfg["end"]]

    target = df[target_col]

    cov_cols = [
        c for c in df.columns
        if c not in [date_col, hour_col, target_col] and not c.startswith("fourier_")
    ]
    covariates = df[cov_cols]

    split_cfg = config.get("split", {})
    train_ratio = split_cfg.get("train_ratio", 0.70)
    val_ratio = split_cfg.get("validation_ratio", 0.15)

    period_start = pd.to_datetime(str(period_cfg.get("start", "2023-01-01")))
    period_end   = pd.to_datetime(str(period_cfg.get("end", "2026-01-01")))
    period_hours = (period_end - period_start).total_seconds() / 3600

    train_hours = int(period_hours * train_ratio)
    val_hours   = int(period_hours * val_ratio)

    VAL_START  = period_start + pd.Timedelta(hours=train_hours)
    TEST_START = period_start + pd.Timedelta(hours=train_hours + val_hours)

    val_start_idx  = int(target.index.searchsorted(VAL_START))
    test_start_idx = int(target.index.searchsorted(TEST_START))

    VAL_START  = target.index[val_start_idx]
    TEST_START = target.index[test_start_idx]

    backtesting_cfg = config.get("backtesting", {})
    FORECAST_HORIZON = backtesting_cfg.get("forecast_horizon", 24)
    STRIDE = backtesting_cfg.get("stride", 24)
    CONTEXT_WINDOW = backtesting_cfg.get("context_window", 24 * 7)

    sarimax_cfg = config.get("sarimax", {})

    _sp_raw = sarimax_cfg.get("seasonal_period", 24)
    if isinstance(_sp_raw, (list, tuple)):
        SEASONAL_PERIOD_OPTIONS = [int(x) for x in _sp_raw]
    else:
        SEASONAL_PERIOD_OPTIONS = [int(_sp_raw)]
    SEASONAL_PERIOD = SEASONAL_PERIOD_OPTIONS[0]
    METHOD = sarimax_cfg.get("method", "CSS")
    MAXITER = sarimax_cfg.get("maxiter", 125)
    INITIALIZATION = sarimax_cfg.get("initialization", "diffuse")
    ENFORCE_STATIONARITY = sarimax_cfg.get("enforce_stationarity", False)
    ENFORCE_INVERTIBILITY = sarimax_cfg.get("enforce_invertibility", False)
    search_cfg = sarimax_cfg.get("search", {})
    fourier_cfg = sarimax_cfg.get("fourier", {})

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

    candidates_cfg = (
        sarimax_cfg.get("features", {}).get("candidates_group")
        or sarimax_cfg.get("features", {}).get("candidates_group")
        or {}
    )

    opt_cfg = config.get("optimization", {})
    N_TRIALS = opt_cfg.get("trials_sarimax", 100)
    STUDY_NAME = opt_cfg.get("study_name", "sarimax_optimization")
    DIRECTION = opt_cfg.get("direction", "minimize")
    TIMEOUT = opt_cfg.get("timeout_seconds", None)

    sampler_type = opt_cfg.get("sampler", "TPESampler")
    seed = config.get("project", {}).get("random_seed", 42)
    n_startup = opt_cfg.get("n_startup_trials", 30)
    if sampler_type == "TPESampler":
        sampler = optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=n_startup,
            multivariate=True,
        )
    else:
        sampler = None

    pruner_type = opt_cfg.get("pruner", "MedianPruner")

    pruner = (
        optuna.pruners.MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=2,
            interval_steps=1,
            n_min_trials=3,
        )
        if pruner_type == "MedianPruner"
        else None
    )

    n_train = len(target[target.index < VAL_START])
    n_val = len(target[(target.index >= VAL_START) & (target.index < TEST_START)])
    n_windows = (n_val - FORECAST_HORIZON) // STRIDE + 1

    print(f"Train: {n_train} obs (hasta {VAL_START})")
    print(f"Val: {n_val} obs (desde {VAL_START} hasta {TEST_START})")
    print(  f"Walk-forward (Transversal): horizon={FORECAST_HORIZON}h, stride={STRIDE}h, context_window={CONTEXT_WINDOW}h (~{n_windows} ventanas)")
    print(f"ARIMA (statsforecast) Params: seasonal_period={SEASONAL_PERIOD_OPTIONS} (method={METHOD})")
    print(f"Fixed features: {fixed_features}")
    print(f"Candidate groups: {list(candidates_cfg.keys())}")
    print()

    logger.info(
        f"Iniciando optimización {STUDY_NAME} ({N_TRIALS} trials): "
        f"STRIDE={STRIDE}h, FORECAST_HORIZON={FORECAST_HORIZON}h, CONTEXT_WINDOW={CONTEXT_WINDOW}h, VAL_START={VAL_START}"
    )

    objective_fn = build_objective(
        target=target,
        covariates=covariates,

        val_start_idx=val_start_idx,
        test_start_idx=test_start_idx,
        forecast_horizon=FORECAST_HORIZON,
        stride=STRIDE,
        context_window=CONTEXT_WINDOW,

        maxiter=MAXITER,
        seasonal_period=SEASONAL_PERIOD,
        seasonal_period_options=SEASONAL_PERIOD_OPTIONS,
        method=METHOD,
        initialization=INITIALIZATION,
        enforce_stationarity=ENFORCE_STATIONARITY,
        enforce_invertibility=ENFORCE_INVERTIBILITY,
        search_cfg=search_cfg,
        fixed_features=fixed_features,
        candidates_cfg=candidates_cfg,
        fourier_cfg=fourier_cfg,
    )

    def progress_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        value = "inf" if trial.value == float("inf") else f"{trial.value:.4f}"
        fit_s = trial.user_attrs.get("fit_seconds", "?")
        wf_s = trial.user_attrs.get("walkforward_seconds", "?")
        n_win = trial.user_attrs.get("n_windows", "?")
        val_mae = trial.user_attrs.get("val_mae")
        test_rmse = trial.user_attrs.get("test_rmse")
        

        p24 = trial.user_attrs.get("lb_pvalue_24")
        p168 = trial.user_attrs.get("lb_pvalue_168")
        p24_s = f"{p24:.2e}" if isinstance(p24, float) else "?"
        p168_s = f"{p168:.2e}" if isinstance(p168, float) else "?"

        val_mae_s = "?" if val_mae is None else f"{val_mae:>10.4f}"
        test_rmse_s = "?" if test_rmse is None else f"{test_rmse:>10.4f}"
        acf24 = trial.user_attrs.get("acf_24")
        acf168 = trial.user_attrs.get("acf_168")
        acf24_s = "?" if acf24 is None else f"{acf24:.4f}"
        acf168_s = "?" if acf168 is None else f"{acf168:.4f}"
        aic = trial.user_attrs.get("aic")
        bic = trial.user_attrs.get("bic")
        aic_s = "?" if aic is None else f"{aic:.2f}"
        bic_s = "?" if bic is None else f"{bic:.2f}"
        best = "inf" if study.best_value == float("inf") else f"{study.best_value:.4f}"
        print(f"[trial {trial.number:>3d}] RMSE={value:>10} | val_mae={val_mae_s} | test_rmse={test_rmse_s} | acf24={acf24_s} | acf168={acf168_s} | aic={aic_s} | bic={bic_s} | fit={fit_s:>6}s | wf={wf_s:>6}s ({n_win} win) | best={best}")

    print(f"Iniciando optimización: {N_TRIALS} trials, n_jobs=4")
    print("=" * 80)

    def _study_is_compatible(study: optuna.Study) -> bool:
        try:
            for trial in study.trials:
                sp = trial.params.get("seasonal_period")
                if sp is not None and sp not in SEASONAL_PERIOD_OPTIONS:
                    return False
            return True
        except Exception:
            return False

    if optuna.storages.get_storage(storage_name) is not None:
        try:
            existing = optuna.load_study(
                study_name=STUDY_NAME,
                storage=storage_name,
            )
            if not _study_is_compatible(existing):
                print(
                    f"⚠ El estudio almacenado '{STUDY_NAME}' contiene trials incompatibles "
                    f"con el espacio de búsqueda actual (seasonal_period ∈ {SEASONAL_PERIOD_OPTIONS}). "
                    "Se eliminará y se iniciará una optimización nueva."
                )
                logger.warning(
                    "Estudio almacenado incompatible con el espacio de búsqueda actual; "
                    "se elimina y se reinicia la optimización."
                )
                optuna.delete_study(study_name=STUDY_NAME, storage=storage_name)
        except Exception as e:
            logger.warning(f"No se pudo validar el estudio almacenado: {e}")

    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction=DIRECTION,
        sampler=sampler,
        pruner=pruner,
        storage=storage_name,
        load_if_exists=True,
    )
    study.optimize(
        objective_fn,
        n_trials=N_TRIALS,
        timeout=TIMEOUT,
        n_jobs=4,
        callbacks=[progress_callback],
        gc_after_trial=True
    )

    logger.info(f"Optimización completada. Best RMSE: {study.best_value:.4f}")

    trials_data = []
    candidate_flags = [g.get("flag", f"use_{k}") for k, g in candidates_cfg.items()]

    p_bounds = search_cfg.get("p", [0])
    d_bounds = search_cfg.get("d", [0])
    q_bounds = search_cfg.get("q", [0])
    P_bounds = search_cfg.get("P", [0])
    D_bounds = search_cfg.get("D", [0])
    Q_bounds = search_cfg.get("Q", [0])

    for trial in study.trials:
        t_data = {
            "trial_number": trial.number,
            "value": trial.value if trial.value != float("inf") else "inf",
            "state": trial.state.name,
            "p": trial.params.get("p", p_bounds[0] if len(p_bounds) == 1 else None),
            "d": trial.params.get("d", d_bounds[0] if len(d_bounds) == 1 else None),
            "q": trial.params.get("q", q_bounds[0] if len(q_bounds) == 1 else None),
            "P": trial.params.get("P", P_bounds[0] if len(P_bounds) == 1 else None),
            "D": trial.params.get("D", D_bounds[0] if len(D_bounds) == 1 else None),
            "Q": trial.params.get("Q", Q_bounds[0] if len(Q_bounds) == 1 else None),
            "k_daily": trial.params.get("k_daily"),
            "k_weekly": trial.params.get("k_weekly"),
            "k_yearly": trial.params.get("k_yearly"),
        }
        for flag in candidate_flags:
            t_data[flag] = trial.params.get(flag)
        t_data["fit_seconds"] = trial.user_attrs.get("fit_seconds")
        t_data["walkforward_seconds"] = trial.user_attrs.get("walkforward_seconds")
        t_data["n_windows"] = trial.user_attrs.get("n_windows")

        for m in ["mae", "rmse", "mape", "smape", "r2"]:
            t_data[f"val_{m}"] = trial.user_attrs.get(f"val_{m}")
            t_data[f"test_{m}"] = trial.user_attrs.get(f"test_{m}")
        t_data["n_windows_test"] = trial.user_attrs.get("n_windows_test")

        t_data["seasonal_period"] = trial.params.get("seasonal_period")
        t_data["acf_24"] = trial.user_attrs.get("acf_24")
        t_data["acf_168"] = trial.user_attrs.get("acf_168")
        t_data["aic"] = trial.user_attrs.get("aic")
        t_data["bic"] = trial.user_attrs.get("bic")
        trials_data.append(t_data)

    trials_df = pd.DataFrame(trials_data)
    csv_output = OUTPUT_DIR / "sarimax_trials.csv"
    trials_df.to_csv(csv_output, index=False)
    print(f"\n✓ Trials exportados a: {csv_output}")
    logger.info(f"Trials exportados a: {csv_output}")

    trials_df_std = study.trials_dataframe()
    csv_output_std = OUTPUT_DIR / "sarimax_trials.csv"
    trials_df_std.to_csv(csv_output_std, index=False)
    print(f"✓ Trials (dataframe estándar) exportados a: {csv_output_std}")
    logger.info(f"Trials (dataframe estándar) exportados a: {csv_output_std}")

    print("\n" + "=" * 80)
    print("RESULTADOS FINALES")
    print("=" * 80)

    best = study.best_trial
    if best.value == float("inf"):
        print("No se encontró ningún trial exitoso.")
        logger.error("No se encontró ningún trial exitoso")
        sys.exit(1)

    best_p = best.params.get("p", p_bounds[0])
    best_d = best.params.get("d", d_bounds[0])
    best_q = best.params.get("q", q_bounds[0])
    best_P = best.params.get("P", P_bounds[0])
    best_D = best.params.get("D", D_bounds[0])
    best_Q = best.params.get("Q", Q_bounds[0])
    best_s = best.params.get("seasonal_period", SEASONAL_PERIOD)

    best_candidate_features = best.user_attrs.get("selected_candidate_features", [])

    print(f"\n Best RMSE: {study.best_value:.4f} ({best.user_attrs.get('n_windows', '?')} ventanas)")
    print(f"  Fit time: {best.user_attrs.get('fit_seconds', '?')}s")
    print(f"  Walk-forward time: {best.user_attrs.get('walkforward_seconds', '?')}s")

    print(f"\n--- Métricas de validación (mejor trial) ---")
    print(f"  mae   = {best.user_attrs.get('val_mae',   float('nan')):.4f}")
    print(f"  rmse  = {study.best_value:.4f}")
    print(f"  mape  = {best.user_attrs.get('val_mape',  float('nan')):.4f}")
    print(f"  smape = {best.user_attrs.get('val_smape', float('nan')):.4f}")
    print(f"  r2    = {best.user_attrs.get('val_r2',    float('nan')):.4f}")

    if best.user_attrs.get("test_rmse") is not None:
        print(f"\n--- Métricas de test por trial (walk-forward, params de train sin refit) ---")
        print(f"  mae   = {best.user_attrs.get('test_mae',   float('nan')):.4f}")
        print(f"  rmse  = {best.user_attrs.get('test_rmse',  float('nan')):.4f}")
        print(f"  mape  = {best.user_attrs.get('test_mape',  float('nan')):.4f}")
        print(f"  smape = {best.user_attrs.get('test_smape', float('nan')):.4f}")
        print(f"  r2    = {best.user_attrs.get('test_r2',    float('nan')):.4f}")
        print(f"  ({best.user_attrs.get('n_windows_test', '?')} ventanas)")

    print(f"\n--- order(p,d,q) ---")
    print(f"  ({best_p}, {best_d}, {best_q})")

    print(f"\n--- seasonal_order(P,D,Q,s) ---")
    print(f"  ({best_P}, {best_D}, {best_Q}, {best_s})")

    print(f"\n--- Fourier K (k_daily, k_weekly, k_yearly) ---")
    print(f"  ({best.params.get('k_daily', '?')}, {best.params.get('k_weekly', '?')}, {best.params.get('k_yearly', '?')})")

    print(f"\n--- Métricas de diagnóstico (sin penalización en el score) ---")
    acf24 = best.user_attrs.get("acf_24")
    acf168 = best.user_attrs.get("acf_168")
    aic = best.user_attrs.get("aic")
    bic = best.user_attrs.get("bic")
    print(f"  |ACF| (lag 24)  = {acf24 if acf24 is None else f'{acf24:.4f}'}")
    print(f"  |ACF| (lag 168) = {acf168 if acf168 is None else f'{acf168:.4f}'}")
    print(f"  AIC = {aic if aic is None else f'{aic:.2f}'}")
    print(f"  BIC = {bic if bic is None else f'{bic:.2f}'}")

    print(f"\n--- Exogenous variables ---")
    for feat in best.user_attrs.get("features", []):
        print(f"  • {feat}")

    best_features = [f for f in best.user_attrs.get("features", []) if f in covariates.columns]

    print("\n" + "=" * 80)
    print("EVALUACIÓN FINAL SOBRE TEST (holdout nunca visto durante la búsqueda)")
    print("=" * 80)

    test_metrics = None
    try:

        best_k_daily = best.params.get("k_daily", 1)
        best_k_weekly = best.params.get("k_weekly", 1)
        best_k_yearly = best.params.get("k_yearly", 1)
        fourier_periods = fourier_cfg.get("periods", {"daily": 24.0, "weekly": 168.0, "yearly": 8765.82})
        best_fourier_df = _build_fourier_df(target.index, fourier_periods, best_k_daily, best_k_weekly, best_k_yearly)

        y_all = target.to_numpy(dtype=float)
        X_all_final = np.hstack(
            [covariates[best_features].to_numpy(dtype=float), best_fourier_df.to_numpy(dtype=float)]
        )

        best_order = (best_p, best_d, best_q)
        best_seasonal_order = (best_P, best_D, best_Q)

        test_start_idx = target.index.get_loc(TEST_START)
        trainval_y = y_all[:test_start_idx]
        trainval_X = X_all_final[:test_start_idx]

        print("Reentrenando el mejor modelo ARIMA (statsforecast) con train + val...")
        test_model = ARIMA(order=best_order, season_length=best_s, seasonal_order=best_seasonal_order, method=METHOD)
        test_model.fit(trainval_y, trainval_X)
        fixed = {k: float(v) for k, v in test_model.model_.get("coef", {}).items()}
        del test_model
        gc.collect()

        test_target = target[target.index >= TEST_START]
        n_test = len(test_target)
        n_windows_test = (n_test - FORECAST_HORIZON) // STRIDE + 1

        print(f"Ejecutando walk-forward en test: horizon={FORECAST_HORIZON}h, stride={STRIDE}h, context_window={CONTEXT_WINDOW}h ({n_windows_test} ventanas)...")
        start_idx = target.index.get_loc(TEST_START)
        all_actuals = []
        all_forecasts = []

        for w in range(n_windows_test):
            abs_pos = start_idx + w * STRIDE
            if abs_pos + FORECAST_HORIZON > len(target):
                break

            all_actuals.append(y_all[abs_pos : abs_pos + FORECAST_HORIZON])
            all_forecasts.append(
                _fixed_window_forecast(
                    y_all, X_all_final, abs_pos, CONTEXT_WINDOW, FORECAST_HORIZON,
                    best_order, best_seasonal_order, best_s, fixed, METHOD,
                )
            )

        if all_actuals:
            y_true = np.concatenate(all_actuals)
            y_pred = np.concatenate(all_forecasts)
            test_metrics = compute_metrics(y_true, y_pred)
            print(f"\nMétricas en test:")
            for k, v in test_metrics.items():
                print(f"  {k:6s} = {v:.4f}")
            logger.info(f"Métricas test ARIMA (statsforecast): {test_metrics}")
        else:
            print("No se pudieron generar ventanas para test.")
            logger.warning("No se pudieron generar ventanas para test.")

    except Exception as e:
        print(f"Error evaluando sobre test: {e}")
        logger.error(f"Error evaluando sobre test ARIMA (statsforecast): {e}")

    config["sarimax"]["best"] = {
        "p": int(best_p),
        "d": int(best_d),
        "q": int(best_q),
        "P": int(best_P),
        "D": int(best_D),
        "Q": int(best_Q),
        "seasonal_period": int(best_s),
        "k_daily": int(best.params.get("k_daily", 1)),
        "k_weekly": int(best.params.get("k_weekly", 1)),
        "k_yearly": int(best.params.get("k_yearly", 1)),
        "features": {
            "fixed": fixed_features,
            "selected": best_candidate_features,
        },

        "rmse": float(round(best.value, 4)),
        "mae": _round_metric(best.user_attrs.get("val_mae")),
        "mape": _round_metric(best.user_attrs.get("val_mape")),
        "smape": _round_metric(best.user_attrs.get("val_smape")),
        "r2": _round_metric(best.user_attrs.get("val_r2")),

        "rmse_test": float(round(test_metrics["rmse"], 4)) if test_metrics else None,
        "mae_test": float(round(test_metrics["mae"], 4)) if test_metrics else None,
        "mape_test": float(round(test_metrics["mape"], 4)) if test_metrics else None,
        "smape_test": float(round(test_metrics["smape"], 4)) if test_metrics else None,
        "r2_test": float(round(test_metrics["r2"], 4)) if test_metrics else None,

        "acf_24": best.user_attrs.get("acf_24"),
        "acf_168": best.user_attrs.get("acf_168"),
        "aic": best.user_attrs.get("aic"),
        "bic": best.user_attrs.get("bic"),
        "trial_number": int(best.number),
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False, allow_unicode=True)

    print(f"\n✓ Mejor configuración guardada en 'sarimax.best' dentro de {CONFIG_PATH}")
    logger.info(f"Mejor configuración guardada en 'sarimax.best' dentro de {CONFIG_PATH}")

    n_failed = sum(1 for t in study.trials if t.value == float("inf"))
    n_pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    n_completed = len(study.trials) - n_failed - n_pruned

    print(f"\n--- Statistics ---")
    print(f"  Completed: {n_completed}/{len(study.trials)}")
    print(f"  Failed (no convergence): {n_failed}")
    print(f"  Pruned (early stopped): {n_pruned}")
    print(f"  Success rate: {100 * n_completed / len(study.trials):.1f}%")

    try:
        importances = optuna.importance.get_param_importances(study)
        print(f"\n--- Parameter Importance ---")
        for k, v in sorted(importances.items(), key=lambda x: -x[1]):
            print(f"  {k:25s} {v:.3f}")
        logger.info(f"Parameter importance calculated: {dict(sorted(importances.items(), key=lambda x: -x[1]))}")
    except Exception as e:
        logger.warning(f"Could not calculate parameter importance: {e}")

    print(f"\n--- Top 10 Trials ---")
    completed = [t for t in study.trials if t.value is not None and t.value != float("inf")]
    completed.sort(key=lambda t: t.value)

    for i, t in enumerate(completed[:10], 1):
        order = t.user_attrs.get("order")
        seasonal = t.user_attrs.get("seasonal_order")
        features = t.user_attrs.get("features", [])
        print(f"\n  [{i}] RMSE={t.value:.4f}")
        print(f"      Order: {order}")
        print(f"      Seasonal: {seasonal}")
        print(f"      Features: {', '.join(features)}")

    logger.info(f"Best trial #{best.number}: RMSE={study.best_value:.4f}, Order=({best_p},{best_d},{best_q}), Seasonal=({best_P},{best_D},{best_Q},{best_s})")

    print("\n" + "=" * 80)
