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
import optuna
from statsmodels.tsa.stattools import acf
from ruamel.yaml import YAML
from darts import TimeSeries
from darts.models import LightGBMModel

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

def load_lightgbm_features(config: dict) -> dict:
    """Extrae la config de features desde config.yaml -> lightgbm.features."""
    feats = config.get("lightgbm", {}).get("features", {})

    past_covariates   = feats.get("past_covariates") or []
    future_covariates = feats.get("future_covariates") or []
    past_groups       = feats.get("past_groups") or {}
    future_groups     = feats.get("future_groups") or {}
    lags_target       = feats.get("lags_target", [-1, -24])
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

def fourier_harmonics(cols) -> dict:
    """Número de armónicos de Fourier efectivamente usado en cada período.

    En LightGBM el grupo de armónicos se activa completo (todo o nada), por lo
    que k no es un hiperparámetro muestreado por Optuna: queda fijado por la
    ingeniería de características, es decir, por las columnas presentes en el
    dataset. Esta función lo recupera para que cada trial quede registrado con
    el k realmente utilizado, y no solo con la marca de uso del grupo.
    """
    ks = {period: 0 for period in FOURIER_PERIODS}
    for col in cols:
        match = FOURIER_COL_RE.match(str(col))
        if match:
            period, k = match.group(1), int(match.group(3))
            ks[period] = max(ks[period], k)
    return ks

def build_objective(
    series: TimeSeries,
    past_covs: TimeSeries,
    future_covs: TimeSeries,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
    test_start: pd.Timestamp,
    forecast_horizon: int,
    stride: int,
    lgb_search_cfg: dict,
    random_seed: int,
    lags_target: list,
    lags_past_covs: list,
    lags_future_covs: list,
    candidate_past_groups: dict,
    candidate_future_groups: dict,
):
    def get_bounds(name, default):
        b = lgb_search_cfg.get(name, default)
        return b if isinstance(b, list) and len(b) == 2 else [b, b]

    lr_bounds           = get_bounds("learning_rate",      [0.005, 0.3])
    leaves_bounds       = get_bounds("num_leaves",         [16, 256])
    depth_bounds        = get_bounds("max_depth",          [3, 12])
    min_child_bounds    = get_bounds("min_child_samples",  [5, 100])
    feat_frac_bounds    = get_bounds("feature_fraction",   [0.6, 1.0])
    bag_frac_bounds     = get_bounds("bagging_fraction",   [0.6, 1.0])
    l1_bounds           = get_bounds("lambda_l1",          [0.0, 5.0])
    l2_bounds           = get_bounds("lambda_l2",          [0.0, 5.0])
    n_est_bounds        = get_bounds("n_estimators",       [100, 1000])

    def objective(trial: optuna.Trial) -> float:
        learning_rate      = trial.suggest_float("learning_rate",    lr_bounds[0],         lr_bounds[1],         log=True)
        num_leaves         = trial.suggest_int("num_leaves",         leaves_bounds[0],     leaves_bounds[1])
        max_depth          = trial.suggest_int("max_depth",          depth_bounds[0],      depth_bounds[1])
        min_child_samples  = trial.suggest_int("min_child_samples",  min_child_bounds[0],  min_child_bounds[1])
        feature_fraction   = trial.suggest_float("feature_fraction", feat_frac_bounds[0],  feat_frac_bounds[1])
        bagging_fraction   = trial.suggest_float("bagging_fraction", bag_frac_bounds[0],   bag_frac_bounds[1])
        lambda_l1          = trial.suggest_float("lambda_l1",         l1_bounds[0],         l1_bounds[1])
        lambda_l2          = trial.suggest_float("lambda_l2",         l2_bounds[0],         l2_bounds[1])
        n_estimators       = trial.suggest_int("n_estimators",        n_est_bounds[0],      n_est_bounds[1])

        past_group_flags   = {name: trial.suggest_categorical(name, [False, True]) for name in candidate_past_groups}
        future_group_flags = {name: trial.suggest_categorical(name, [False, True]) for name in candidate_future_groups}

        past_groups_sel   = [name for name, on in past_group_flags.items() if on]
        future_groups_sel = [name for name, on in future_group_flags.items() if on]

        past_cols_eff   = [col for name in past_groups_sel for col in candidate_past_groups[name]]
        future_cols_eff = [col for name in future_groups_sel for col in candidate_future_groups[name]]

        fourier_k = fourier_harmonics(past_cols_eff + future_cols_eff)
        for period, k in fourier_k.items():
            trial.set_user_attr(f"fourier_k_{period}", int(k))

        lags_past_covs_eff = lags_past_covs if past_cols_eff else None
        lags_future_eff    = lags_future_covs if future_cols_eff else None

        try:
            t_fit_start = time.time()

            model = LightGBMModel(
                lags=lags_target,
                lags_past_covariates=lags_past_covs_eff,
                lags_future_covariates=lags_future_eff,
                output_chunk_length=1,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                max_depth=max_depth,
                min_child_samples=min_child_samples,
                feature_fraction=feature_fraction,
                bagging_fraction=bagging_fraction,
                lambda_l1=lambda_l1,
                lambda_l2=lambda_l2,
                n_estimators=n_estimators,
                random_state=random_seed,
                verbose=-1,
            )

            train_series = series.drop_after(val_start)
            train_past_covs = (
                past_covs.drop_after(val_start) if past_covs is not None else None
            )
            train_future_covs = (
                future_covs.drop_after(val_start) if future_covs is not None else None
            )

            fit_kwargs = {"series": train_series, "verbose": False}
            if past_cols_eff:
                fit_kwargs["past_covariates"] = train_past_covs[past_cols_eff]
            if future_cols_eff:
                fit_kwargs["future_covariates"] = train_future_covs[future_cols_eff]

            model.fit(**fit_kwargs)
            fit_seconds = time.time() - t_fit_start

            t_wf_start = time.time()
            hf_kwargs = {
                "series": series,
                "start": val_start,
                "forecast_horizon": forecast_horizon,
                "stride": stride,
                "retrain": False,
                "overlap_end": False,
                "last_points_only": False,
                "verbose": False,
            }
            if past_cols_eff:
                hf_kwargs["past_covariates"] = past_covs[past_cols_eff]
            if future_cols_eff:
                hf_kwargs["future_covariates"] = future_covs[future_cols_eff]

            backtest_forecasts = model.historical_forecasts(**hf_kwargs)
            wf_seconds = time.time() - t_wf_start

            rmses         = []
            all_actual    = []
            all_predicted = []
            for fcast in backtest_forecasts:
                if fcast.end_time() >= val_end:
                    continue
                actual = series.slice_intersect(fcast)
                act  = actual.values().flatten()
                pred = fcast.values().flatten()
                err  = act - pred
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
                trial.set_user_attr("n_estimators", int(model.model.n_estimators))
            except Exception:
                trial.set_user_attr("n_estimators", None)
            trial.set_user_attr("num_leaves", trial.params.get("num_leaves"))

            test_metrics = None
            n_windows_test = 0
            try:
                hf_test_kwargs = {
                    "series": series,
                    "start": test_start,
                    "forecast_horizon": forecast_horizon,
                    "stride": stride,
                    "retrain": False,
                    "overlap_end": False,
                    "last_points_only": False,
                    "verbose": False,
                }
                if past_cols_eff:
                    hf_test_kwargs["past_covariates"] = past_covs[past_cols_eff]
                if future_cols_eff:
                    hf_test_kwargs["future_covariates"] = future_covs[future_cols_eff]

                backtest_test = model.historical_forecasts(**hf_test_kwargs)

                test_actual = []
                test_predicted = []
                for fcast in backtest_test:
                    actual = series.slice_intersect(fcast)
                    test_actual.append(actual.values().flatten())
                    test_predicted.append(fcast.values().flatten())

                if test_actual:
                    n_windows_test = len(test_actual)
                    test_metrics = compute_metrics(
                        np.concatenate(test_actual), np.concatenate(test_predicted)
                    )
            except Exception:
                test_metrics = None

            trial.set_user_attr("n_windows",           len(rmses))
            trial.set_user_attr("fit_seconds",         round(fit_seconds, 1))
            trial.set_user_attr("walkforward_seconds", round(wf_seconds, 1))
            trial.set_user_attr("past_groups_selected",   past_groups_sel)
            trial.set_user_attr("future_groups_selected", future_groups_sel)
            trial.set_user_attr("n_past_cols",  len(past_cols_eff))
            trial.set_user_attr("n_future_cols", len(future_cols_eff))
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
    config = load_config(CONFIG_PATH)
    seed   = config.get("project", {}).get("random_seed", 42)
    np.random.seed(seed)

    feats = load_lightgbm_features(config)
    past_covariates        = feats["past_covariates"]
    future_covariates      = feats["future_covariates"]
    candidate_past_groups  = feats["past_groups"]
    candidate_future_groups = feats["future_groups"]
    lags_target            = feats["lags_target"]
    lags_past_covs         = feats["lags_past_covs"]
    lags_future_covs       = feats["lags_future_covs"]

    LOG_DIR    = Path(config.get("output", {}).get("logs", "src/optimization/logs"))
    OUTPUT_DIR = Path(config.get("output", {}).get("optimization", "output/optimization"))
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("lightgbm_optuna")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_DIR / "lightgbm_optimization.log")
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
    period_hours = (period_end - period_start).total_seconds() / 3600

    split_cfg   = config.get("split", {})
    train_ratio = split_cfg.get("train_ratio", 0.70)
    val_ratio   = split_cfg.get("validation_ratio", 0.15)

    train_hours = int(period_hours * train_ratio)
    val_hours   = int(period_hours * val_ratio)

    VAL_START  = period_start + pd.Timedelta(hours=train_hours)
    VAL_END    = period_start + pd.Timedelta(hours=train_hours + val_hours)
    TEST_START = VAL_END

    df_filtered = df.loc[(df.index >= period_start) & (df.index <= period_end)]

    n_train_ref = len(df_filtered[df_filtered.index < VAL_START])
    n_val       = len(df_filtered[(df_filtered.index >= VAL_START) & (df_filtered.index < VAL_END)])
    n_test      = len(df_filtered[df_filtered.index >= TEST_START])
    n_total     = len(df_filtered)

    df = df_filtered

    if target_col not in df.columns:
        logger.error(f"Columna target '{target_col}' no encontrada")
        print(f"Error: columna target '{target_col}' no encontrada")
        sys.exit(1)

    candidate_cols = [c for group in list(candidate_past_groups.values()) + list(candidate_future_groups.values()) for c in group]
    missing = [c for c in candidate_cols if c not in df.columns]
    if missing:
        logger.error(f"Columnas faltantes: {missing}")
        print(f"Advertencia: columnas faltantes: {missing}")

    all_cov_cols = past_covariates + future_covariates
    missing_covs = [c for c in all_cov_cols if c not in df.columns]
    if missing_covs:
        logger.error(f"Columnas covariables faltantes: {missing_covs}")
        print(f"Advertencia: columnas covariables faltantes: {missing_covs}")

    bt_cfg           = config.get("backtesting", {})
    FORECAST_HORIZON = bt_cfg.get("forecast_horizon", 24)
    STRIDE           = bt_cfg.get("stride", 24)
    CONTEXT_WINDOW   = bt_cfg.get("context_window", 24 * 7)

    n_windows = (n_val - FORECAST_HORIZON) // STRIDE + 1 if n_val > FORECAST_HORIZON else 0

    print(f"Train: {n_train_ref} obs (hasta {VAL_START})")
    print(f"Val:   {n_val} obs ({VAL_START} -> {VAL_END})")
    print(f"Test:  {n_test} obs (desde {TEST_START}, reservado)")
    print(f"Walk-forward: horizon={FORECAST_HORIZON}h, stride={STRIDE}h (~{n_windows} ventanas)")
    print(f"Lags del target: {lags_target}")
    print(f"Lags de past_covariates: {lags_past_covs}")
    print(f"Past groups candidatos: {list(candidate_past_groups)}")
    print(f"Future groups candidatos: {list(candidate_future_groups)}")
    print()

    logger.info(f"Dataset: {n_total} obs | Train={n_train_ref}, Val={n_val}, Test={n_test}")
    logger.info(f"Split: VAL_START={VAL_START}, VAL_END={VAL_END}, TEST_START={TEST_START}")
    logger.info(f"Backtesting: horizon={FORECAST_HORIZON}, stride={STRIDE}, context_window={CONTEXT_WINDOW}")

    series_ts      = TimeSeries.from_dataframe(df[[target_col]], freq="h")
    past_covs_ts   = (
        TimeSeries.from_dataframe(df[past_covariates], freq="h")
        if past_covariates else None
    )
    future_covs_ts = (
        TimeSeries.from_dataframe(df[future_covariates], freq="h")
        if future_covariates else None
    )

    opt_cfg    = config.get("optimization", {})
    N_TRIALS   = opt_cfg.get("trials_lightgbm", 150)
    STUDY_NAME = opt_cfg.get("study_name", "forecasting") + "_lightgbm"
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

    lgb_cfg        = config.get("lightgbm", {})
    lgb_search_cfg = lgb_cfg.get("search", {})

    objective_fn = build_objective(
        series=series_ts,
        past_covs=past_covs_ts,
        future_covs=future_covs_ts,
        val_start=VAL_START,
        val_end=VAL_END,
        test_start=TEST_START,
        forecast_horizon=FORECAST_HORIZON,
        stride=STRIDE,
        lgb_search_cfg=lgb_search_cfg,
        random_seed=seed,
        lags_target=lags_target,
        lags_past_covs=lags_past_covs,
        lags_future_covs=lags_future_covs,
        candidate_past_groups=candidate_past_groups,
        candidate_future_groups=candidate_future_groups,
    )

    def progress_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        value = "inf" if trial.value == float("inf") else f"{trial.value:.4f}"
        fit_s = trial.user_attrs.get("fit_seconds", "?")
        wf_s  = trial.user_attrs.get("walkforward_seconds", "?")
        n_win = trial.user_attrs.get("n_windows", "?")
        state = trial.state.name
        best  = "inf" if study.best_value == float("inf") else f"{study.best_value:.4f}"
        p_sel = trial.user_attrs.get("past_groups_selected", [])
        f_sel = trial.user_attrs.get("future_groups_selected", [])
        groups_txt = f"past={p_sel} future={f_sel}" if (p_sel or f_sel) else "past=[] future=[]"
        fourier_k = {p: trial.user_attrs.get(f"fourier_k_{p}") for p in FOURIER_PERIODS}
        k_txt    = "/".join(str(fourier_k[p]) for p in FOURIER_PERIODS)
        val_mae = trial.user_attrs.get("val_mae")
        test_rmse = trial.user_attrs.get("test_rmse")
        val_mae_s = "?" if val_mae is None else f"{val_mae:>10.4f}"
        test_rmse_s = "?" if test_rmse is None else f"{test_rmse:>10.4f}"
        acf24 = trial.user_attrs.get("acf_24")
        acf168 = trial.user_attrs.get("acf_168")
        acf24_s = "?" if acf24 is None else f"{acf24:.4f}"
        acf168_s = "?" if acf168 is None else f"{acf168:.4f}"
        print(f"[trial {trial.number:>3d}] RMSE={value:>10} | val_mae={val_mae_s} | test_rmse={test_rmse_s} | acf24={acf24_s} | acf168={acf168_s} | fit={fit_s:>6}s | wf={wf_s:>6}s ({n_win} win) | {state:8s} | best={best} | {groups_txt} | k_fourier={k_txt}")

    print(f"Iniciando optimización LightGBM: {N_TRIALS} trials, n_jobs={N_JOBS}")
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

    logger.info(f"Optimización LightGBM completada. Best RMSE (val): {study.best_value:.4f}")

    trials_data = []
    for trial in study.trials:
        row = {
            "trial_number":        trial.number,
            "value":               trial.value if trial.value != float("inf") else "inf",
            "state":               trial.state.name,
            "learning_rate":       trial.params.get("learning_rate"),
            "num_leaves":          trial.params.get("num_leaves"),
            "max_depth":           trial.params.get("max_depth"),
            "min_child_samples":   trial.params.get("min_child_samples"),
            "feature_fraction":    trial.params.get("feature_fraction"),
            "bagging_fraction":    trial.params.get("bagging_fraction"),
            "lambda_l1":           trial.params.get("lambda_l1"),
            "lambda_l2":           trial.params.get("lambda_l2"),
            "n_estimators":        trial.params.get("n_estimators"),
            **{g: trial.params.get(g) for g in list(candidate_past_groups) + list(candidate_future_groups)},
            "past_groups_selected":   ",".join(trial.user_attrs.get("past_groups_selected", [])),
            "future_groups_selected": ",".join(trial.user_attrs.get("future_groups_selected", [])),
            **{f"fourier_k_{p}": trial.user_attrs.get(f"fourier_k_{p}") for p in FOURIER_PERIODS},
            "fit_seconds":         trial.user_attrs.get("fit_seconds"),
            "walkforward_seconds": trial.user_attrs.get("walkforward_seconds"),
            "n_windows":           trial.user_attrs.get("n_windows"),
            "n_windows_test":      trial.user_attrs.get("n_windows_test"),
            "error":               trial.user_attrs.get("error"),
            "acf_24":              trial.user_attrs.get("acf_24"),
            "acf_168":             trial.user_attrs.get("acf_168"),
            "residual_mean":       trial.user_attrs.get("residual_mean"),
        }

        for m in ["mae", "rmse", "mape", "smape", "r2"]:
            row[f"val_{m}"]  = trial.user_attrs.get(f"val_{m}")
            row[f"test_{m}"] = trial.user_attrs.get(f"test_{m}")
        trials_data.append(row)

    trials_df  = pd.DataFrame(trials_data)
    csv_output = OUTPUT_DIR / "lightgbm_trials.csv"
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

    best_lr         = best.params.get("learning_rate")
    best_leaves     = best.params.get("num_leaves")
    best_depth      = best.params.get("max_depth")
    best_min_child  = best.params.get("min_child_samples")
    best_feat_frac  = best.params.get("feature_fraction")
    best_bag_frac   = best.params.get("bagging_fraction")
    best_l1         = best.params.get("lambda_l1")
    best_l2         = best.params.get("lambda_l2")
    best_n_est      = best.params.get("n_estimators")

    best_past_groups   = [g for g in candidate_past_groups if best.params.get(g)]
    best_future_groups = [g for g in candidate_future_groups if best.params.get(g)]
    best_past_cols     = [col for g in best_past_groups for col in candidate_past_groups[g]]
    best_future_cols   = [col for g in best_future_groups for col in candidate_future_groups[g]]
    best_fourier_k     = fourier_harmonics(best_past_cols + best_future_cols)

    print(f"\nBest RMSE (val): {study.best_value:.4f} ({best.user_attrs.get('n_windows', '?')} ventanas)")
    print(f"  mae   (val)       = {best.user_attrs.get('val_mae',   float('nan')):.4f}")
    print(f"  rmse  (val)       = {study.best_value:.4f}")
    print(f"  mape  (val)       = {best.user_attrs.get('val_mape',  float('nan')):.4f}")
    print(f"  smape (val)       = {best.user_attrs.get('val_smape', float('nan')):.4f}")
    print(f"  r2    (val)       = {best.user_attrs.get('val_r2',    float('nan')):.4f}")
    print(f"  learning_rate     = {best_lr:.6f}")
    print(f"  num_leaves        = {best_leaves}")
    print(f"  max_depth         = {best_depth}")
    print(f"  min_child_samples = {best_min_child}")
    print(f"  feature_fraction  = {best_feat_frac:.4f}")
    print(f"  bagging_fraction  = {best_bag_frac:.4f}")
    print(f"  lambda_l1         = {best_l1:.4f}")
    print(f"  lambda_l2         = {best_l2:.4f}")
    print(f"  n_estimators      = {best_n_est}")
    print(f"  past_groups       = {best_past_groups}")
    print(f"  future_groups     = {best_future_groups}")
    print(f"  fourier_k (d/w/y) = {best_fourier_k['daily']}/{best_fourier_k['weekly']}/{best_fourier_k['yearly']}")
    print(f"  past_covariates   = {best_past_cols}")
    print(f"  future_covariates = {best_future_cols}")

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
        trainval_series    = series_ts.drop_after(TEST_START)
        trainval_past_covs = (
            past_covs_ts.drop_after(TEST_START) if past_covs_ts is not None else None
        )
        trainval_future_covs = (
            future_covs_ts.drop_after(TEST_START) if future_covs_ts is not None else None
        )

        final_model = LightGBMModel(
            lags=lags_target,
            lags_past_covariates=lags_past_covs if best_past_cols else None,
            lags_future_covariates=lags_future_covs if best_future_cols else None,
            output_chunk_length=1,
            learning_rate=best_lr,
            num_leaves=best_leaves,
            max_depth=best_depth,
            min_child_samples=best_min_child,
            feature_fraction=best_feat_frac,
            bagging_fraction=best_bag_frac,
            lambda_l1=best_l1,
            lambda_l2=best_l2,
            n_estimators=best_n_est,
            random_state=seed,
            verbose=-1,
        )

        print("Reentrenando mejor modelo con train + val...")
        print(f"  past_covariates   = {best_past_cols}")
        print(f"  future_covariates = {best_future_cols}")

        fit_kwargs = {"series": trainval_series, "verbose": False}
        if best_past_cols:
            fit_kwargs["past_covariates"] = trainval_past_covs[best_past_cols]
        if best_future_cols:
            fit_kwargs["future_covariates"] = trainval_future_covs[best_future_cols]
        final_model.fit(**fit_kwargs)

        hf_kwargs = {
            "series": series_ts,
            "start": TEST_START,
            "forecast_horizon": FORECAST_HORIZON,
            "stride": STRIDE,
            "retrain": False,
            "overlap_end": False,
            "last_points_only": False,
            "verbose": False,
        }
        if best_past_cols:
            hf_kwargs["past_covariates"] = past_covs_ts[best_past_cols]
        if best_future_cols:
            hf_kwargs["future_covariates"] = future_covs_ts[best_future_cols]

        backtest_test = final_model.historical_forecasts(**hf_kwargs)

        all_actual    = []
        all_predicted = []
        for fcast in backtest_test:
            actual = series_ts.slice_intersect(fcast)
            all_actual.append(actual.values().flatten())
            all_predicted.append(fcast.values().flatten())

        if all_actual:
            y_true = np.concatenate(all_actual)
            y_pred = np.concatenate(all_predicted)
            test_metrics   = compute_metrics(y_true, y_pred)
            n_windows_test = len(all_actual)
            print(f"\n  Métricas en test ({n_windows_test} ventanas):")
            for k, v in test_metrics.items():
                print(f"    {k:6s} = {v:.4f}")
            logger.info(f"Métricas test LightGBM: {test_metrics}")
        else:
            print("  No se generaron ventanas de test.")
            logger.warning("No se generaron ventanas de test")

    except Exception as e:
        print(f"  Error evaluando sobre test: {e}")
        logger.error(f"Error evaluando sobre test LightGBM: {e}")

    if "lightgbm" not in config:
        config["lightgbm"] = {}
    if "best" not in config["lightgbm"]:
        config["lightgbm"]["best"] = {}

    bs = config["lightgbm"]["best"]
    bs["learning_rate"]      = float(best_lr)
    bs["num_leaves"]         = int(best_leaves)
    bs["max_depth"]          = int(best_depth)
    bs["min_child_samples"]  = int(best_min_child)
    bs["feature_fraction"]   = float(round(best_feat_frac, 4))
    bs["bagging_fraction"]   = float(round(best_bag_frac, 4))
    bs["lambda_l1"]          = float(round(best_l1, 4))
    bs["lambda_l2"]          = float(round(best_l2, 4))
    bs["n_estimators"]       = int(best_n_est)
    bs["lags"]               = lags_target
    bs["lags_past_covariates"]     = lags_past_covs if best_past_cols else None
    bs["lags_future_covariates"]   = lags_future_covs if best_future_cols else None
    bs["past_covariates"]    = best_past_cols
    bs["future_covariates"]  = best_future_cols
    bs["past_groups"]        = best_past_groups
    bs["future_groups"]      = best_future_groups
    bs["fourier_k_daily"]    = best_fourier_k["daily"]
    bs["fourier_k_weekly"]   = best_fourier_k["weekly"]
    bs["fourier_k_yearly"]   = best_fourier_k["yearly"]
    bs["output_chunk_length"]= 1

    def _round_metric(v):
        """Redondea a 4 decimales; None/NaN se guardan como None."""
        if v is None:
            return None
        v = float(v)
        return float(round(v, 4)) if not np.isnan(v) else None

    bs["rmse"]               = _round_metric(best.value)
    bs["mae"]                = _round_metric(best.user_attrs.get("val_mae"))
    bs["mape"]               = _round_metric(best.user_attrs.get("val_mape"))
    bs["smape"]              = _round_metric(best.user_attrs.get("val_smape"))
    bs["r2"]                 = _round_metric(best.user_attrs.get("val_r2"))

    bs["rmse_test"]          = _round_metric(test_metrics["rmse"]  if test_metrics else None)
    bs["mae_test"]           = _round_metric(test_metrics["mae"]   if test_metrics else None)
    bs["mape_test"]          = _round_metric(test_metrics["mape"]  if test_metrics else None)
    bs["smape_test"]         = _round_metric(test_metrics["smape"] if test_metrics else None)
    bs["r2_test"]            = _round_metric(test_metrics["r2"]    if test_metrics else None)
    bs["trial_number"]       = int(best.number)
    bs["datetime"]           = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    save_config(config, CONFIG_PATH)
    print(f"\n  Mejor configuración guardada en 'lightgbm.best' en {CONFIG_PATH}")
    logger.info(f"Guardado en lightgbm.best: {CONFIG_PATH}")

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
        print(f"\n  [{i}] RMSE={t.value:.4f}")
        print(f"      lr={t.params.get('learning_rate', 0):.6f}  leaves={t.params.get('num_leaves')}  "
              f"depth={t.params.get('max_depth')}  min_child={t.params.get('min_child_samples')}")
        print(f"      feat_frac={t.params.get('feature_fraction', 0):.3f}  bag_frac={t.params.get('bagging_fraction', 0):.3f}  "
              f"l1={t.params.get('lambda_l1', 0):.3f}  l2={t.params.get('lambda_l2', 0):.3f}  "
              f"n_est={t.params.get('n_estimators', 0)}")
        t_past_cols   = [c for g in candidate_past_groups   if t.params.get(g) for c in candidate_past_groups[g]]
        t_future_cols = [c for g in candidate_future_groups if t.params.get(g) for c in candidate_future_groups[g]]
        t_k           = fourier_harmonics(t_past_cols + t_future_cols)
        print(f"      past={t_past_cols or '-'}")
        print(f"      future={t_future_cols or '-'}")
        print(f"      k_fourier (d/w/y)={t_k['daily']}/{t_k['weekly']}/{t_k['yearly']}")

    logger.info(f"Best trial LightGBM #{best.number}: RMSE(val)={study.best_value:.4f}, test_metrics={test_metrics}")

    print("\n" + "=" * 80)