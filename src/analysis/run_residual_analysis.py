"""
Análisis de residuos de los modelos evaluados.

Script manual que se ejecuta DESPUES de correr los scripts de evaluation.
Lee los CSVs de predicciones guardados por cada evaluation script y genera
el analisis de residuos completo (tests estadisticos + graficos).

Los graficos se AGRUPAN por horizonte x tipo de test: un solo PNG por
horizonte y tipo de test, con un renglon por modelo (homocedasticidad en
grilla 2x2: dos modelos a la izquierda y dos a la derecha), en lugar de
6 PNG por modelo x horizonte. Con 4 modelos y 4 horizontes salen 16 imagenes en vez
de 96. Archivos generados por horizonte, en output/evaluation/residuals/plots:

    {h}h_residuals_serie.png             serie temporal de residuos
    {h}h_residuals_normalidad.png        histograma + QQ (un renglon por modelo)
    {h}h_residuals_autocorrelacion.png   ACF + PACF
    {h}h_residuals_homocedasticidad.png  residuos vs predichos (grilla 2x2)

Uso:
    python src/analysis/run_residual_analysis.py                         # Todos los modelos
    python src/analysis/run_residual_analysis.py --model sarimax         # Solo SARIMAX
    python src/analysis/run_residual_analysis.py --horizon 24            # Solo 24h
    python src/analysis/run_residual_analysis.py --model lightgbm --horizon 3  # Combinado
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import argparse
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
from ruamel.yaml import YAML

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.residual_analysis import (
    run_residual_analysis,
    compute_residuals,
    save_grouped_plots,
    dump_residual_yaml,
)
from evaluation.forecast_meta import read_forecast_meta

EVAL_DIR = Path("output/evaluation")
OUTPUT_DIR = EVAL_DIR / "residuals"
CONFIG_PATH = Path("config.yaml")

AVAILABLE_MODELS = ["sarimax", "lightgbm", "lstm", "chronos2"]
AVAILABLE_HORIZONS = [24, 3, 6, 12]

def load_config_stride():
    """Lee backtesting.stride de config.yaml.

    Solo se usa como FALLBACK cuando el CSV no tiene sidecar. La fuente
    reliable es el ``<csv>.meta.json`` que escribe el evaluation script; ver
    ``resolve_grid``.
    """
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = YAML(typ="safe").load(f) or {}
        return cfg.get("backtesting", {}).get("stride")
    except Exception as e:
        print(f"⚠  No se pudo leer stride de {CONFIG_PATH}: {e}")
        return None

def resolve_grid(csv_path, horizon, config_stride):
    """Determina el (horizon, stride) REAL con el que se genero el CSV.

    Prioridad:
      1. El sidecar ``<csv>.meta.json``, que escribe el evaluation script con
         el stride que efectivamente uso. Es la fuente fiable.
      2. ``config.yaml`` como fallback, para CSVs de corridas antiguas sin
         sidecar. Se avisa por pantalla porque puede ser incorrecto.

    Sin esta resolucion, un CSV valido generado con ``stride == horizon``
    (ventanas contiguas) seria rechazado cuando el config dice otro stride, y
    uno fabricado pasaria el chequeo si el config coincide por casualidad.
    """
    #OJO:el stride real sale del sidecar; la config queda de respaldo.
    meta = read_forecast_meta(csv_path)
    if meta:
        return meta["horizon"], meta["stride"], f"{csv_path.name}.meta.json"

    if config_stride is None:
        return horizon, None, "config.yaml (sin stride)"

    print(f"  ⚠  {csv_path.name} sin sidecar .meta.json: se asume stride="
          f"{config_stride} de config.yaml. Puede no ser el usado realmente; "
          f"re-corra el evaluation script para registrarlo.")
    return horizon, int(config_stride), "config.yaml (fallback, sin sidecar)"

def find_predictions_csv(model, horizon):
    """Busca el CSV de predicciones del test para un modelo y horizonte."""
    csv_path = EVAL_DIR / f"{model}_test_predictions_{horizon}h.csv"
    if csv_path.exists():
        return csv_path
    return None

def load_predictions(csv_path):
    """Carga un CSV de predicciones y retorna actual, predicted, datetime_index."""
    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df["actual"].values, df["predicted"].values, df["datetime"]

def print_residual_summary(results):
    """Imprime un resumen legible de los resultados del análisis."""
    n = results["n_observations"]
    stats = results["residual_stats"]
    norm = results["normalidad"]
    acorr = results["autocorrelacion"]
    het = results["homocedasticidad"]
    biases = results["sesgos_sistematicos"]

    print(f"\n  Observaciones: {n}")
    print(f"  RMSE:          {stats['rmse']:.4f} MW")
    print(f"  Bias medio:    {stats['mean']:.4f} MW")
    print(f"  Std residuos:  {stats['std']:.4f} MW")
    print(f"  Rango:         [{stats['min']:.2f}, {stats['max']:.2f}] MW")

    grid = results.get("forecast_grid", {})
    if grid.get("forecast_grid_validated"):
        src = grid.get("stride_source")
        sufijo = f"  [fuente: {src}]" if src else ""
        print(f"\n  Grilla de forecast: ✅ verificada "
              f"(horizon={grid['horizon']}, stride={grid['stride']}, "
              f"gap={grid['gap_hours']}h, {grid['n_windows']} ventanas, "
              f"span={grid['span_hours']}h){sufijo}")
    else:
        print(f"\n  Grilla de forecast: ⚠  NO verificada "
              f"({grid.get('reason', 'sin stride en config')})")

    eff = results.get("tamano_efecto", {})
    print(f"\n  Normalidad (Shapiro-Wilk): p={norm['shapiro_pval']:.3e} "
          f"-> H0 {'no se rechaza' if norm['shapiro_normal'] else 'rechazada'}")
    if eff.get("n"):
        print(f"    sesgo      {eff['skewness']:>8.3f}  ({eff['skewness_magnitude']})")
        print(f"    kurtosis   {eff['excess_kurtosis']:>8.2f}  ({eff['kurtosis_magnitude']})")
        print(f"    >2sigma    {eff['pct_beyond_2sigma']:>7.2f}%  (normal: 4.55%)")
        print(f"    >3sigma    {eff['pct_beyond_3sigma']:>7.2f}%  (normal: 0.27%)")
        print(f"    max|z|     {eff['max_abs_zscore']:>7.2f}")
        print(f"    peor 1% aporta {eff['worst_1pct_share_of_mse']:.1f}% del error cuadrático")
        if eff["kurtosis_magnitude"] in ("despreciable", "leve"):
            print(f"    VEREDICTO: desviación NO material (rechazo por potencia)")
        else:
            print(f"    VEREDICTO: desviación MATERIAL (cola pesada genuina)")
    if norm.get("conclusion"):
        print(f"    {norm['conclusion']}")

    print(f"\n  Autocorrelación:")
    print(f"    ACF lag 1:   {acorr['acf_lag_1']:.4f}  "
          f"({acorr.get('acf1_variance_explained_pct', float('nan')):.1f}% de la "
          f"varianza residual, {acorr.get('acf1_magnitude', 'n/a')})")
    print(f"    ACF lag 24:  {acorr['acf_lag_24']:.4f}")
    if acorr.get("acf_lag_168") is not None:
        print(f"    ACF lag 168: {acorr['acf_lag_168']:.4f}")
    else:
        max_lag = acorr.get("acf_max_lag_computed", 0)
        print(f"    ACF lag 168: n/a (n={n} → techo n//2-1={max_lag}; "
              f"se necesitan n>=338 para medir la semana)")
    print(f"    PACF lag 1:  {acorr['pacf_lag_1']:.4f}")
    print(f"    PACF lag 24: {acorr['pacf_lag_24']:.4f}")
    if acorr.get("ljung_box_pval") is not None:
        print(f"    Ljung-Box({acorr.get('ljung_box_lags')}): "
              f"Q={acorr['ljung_box_stat']:.2f}, p={acorr['ljung_box_pval']:.3e}")

    sp_corr = het["spearman_corr"]
    sp_mag = ("despreciable" if abs(sp_corr) < 0.1 else
              "leve" if abs(sp_corr) < 0.3 else
              "moderada" if abs(sp_corr) < 0.5 else "severa")
    print(f"\n  Homocedasticidad:")
    print(f"    Spearman: |residuo|~predicción  r={sp_corr:+.4f} ({sp_mag}), "
          f"p={het['spearman_pval']:.3e}")

    gq_p = het.get("goldfeld_quandt_by_prediction_pval")
    if gq_p is None:
        print(f"    GQ por predicción:   n/a ({het.get('goldfeld_quandt_note', '')})")
    else:
        gq = "estable" if het["goldfeld_quandt_by_prediction_homoscedastic"] \
            else "la varianza escala con la predicción"
        print(f"    GQ por predicción:   {gq} (p={gq_p:.3e})")

    gq_t = het.get("goldfeld_quandt_by_time_pval")
    if gq_t is not None:
        gqt = "estable" if het["goldfeld_quandt_by_time_stable_variance"] \
            else "la varianza cambia a lo largo del período"
        print(f"    GQ por tiempo:       {gqt} (p={gq_t:.3e})")

    mae_hour = biases["mae_by_hour"]
    if mae_hour:
        max_mae = max(mae_hour.values()) if mae_hour.values() else 1
        print(f"\n  MAE por hora del día:")
        for h in sorted(mae_hour.keys()):
            bar = "█" * int(mae_hour[h] / max_mae * 20) if max_mae > 0 else ""
            print(f"    {int(h):02d}h: {mae_hour[h]:6.2f} MW  {bar}")

    mae_season = biases["mae_by_season"]
    if mae_season:
        print(f"\n  MAE por estación:")
        for s in ["verano", "otoño", "invierno", "primavera"]:
            if s in mae_season:
                print(f"    {s:10s}: {mae_season[s]:6.2f} MW")

def main():
    parser = argparse.ArgumentParser(
        description="Análisis de residuos de modelos de forecasting"
    )
    parser.add_argument(
        "--model", choices=AVAILABLE_MODELS + ["all"], default="all",
        help="Modelo a analizar (default: all)"
    )
    parser.add_argument(
        "--horizon", type=int, choices=AVAILABLE_HORIZONS, default=None,
        help="Horizonte en horas (default: detectar de CSVs disponibles)"
    )
    args = parser.parse_args()

    models = AVAILABLE_MODELS if args.model == "all" else [args.model]

    if args.horizon is not None:
        horizons = [args.horizon]
    else:
        horizons = []
        for h in AVAILABLE_HORIZONS:
            for m in models:
                if find_predictions_csv(m, h) is not None:
                    horizons.append(h)
                    break
        horizons = sorted(set(horizons)) or [24]

    config_stride = load_config_stride()
    print("=" * 80)
    print("ANÁLISIS DE RESIDUOS")
    print(f"Modelos: {', '.join(models)}")
    print(f"Horizontes: {', '.join(str(h) + 'h' for h in horizons)}")
    print(f"Stride (config, solo fallback): {config_stride}")
    print(f"Salida: {OUTPUT_DIR}")
    print("=" * 80)

    total_analyzed = 0
    total_skipped = 0

    for horizon in horizons:

        entries = []
        yaml_paths = {}

        for model in models:
            csv_path = find_predictions_csv(model, horizon)
            if csv_path is None:
                print(f"\n⚠  {model} {horizon}h: no se encontró {EVAL_DIR / f'{model}_test_predictions_{horizon}h.csv'}")
                print(f"   Ejecutá primero: python src/evaluation/{model}_evaluation.py")
                continue

            print(f"\n{'─' * 80}")
            print(f"  {model.upper()} — {horizon}h")
            print(f"{'─' * 80}")

            actual, predicted, datetime_index = load_predictions(csv_path)
            print(f"  CSV: {csv_path}")

            real_h, real_stride, grid_src = resolve_grid(csv_path, horizon, config_stride)
            if real_h != horizon:
                print(f"  ⚠  sidecar declara horizon={real_h} pero el archivo se "
                      f"llama {horizon}h. Se usa horizon={real_h}.")

            try:
                results = run_residual_analysis(
                    actual=actual,
                    predicted=predicted,
                    datetime_index=datetime_index,
                    model_name=model,
                    horizon=real_h,
                    output_dir=OUTPUT_DIR,
                    stride=real_stride,
                    save_plots=False,   
                    save_yaml=False,    
                )
            except ValueError as e:

                print(f"\n  ✗ OMITIDO — CSV no analizable:")
                print(f"    {e}")
                total_skipped += 1
                continue

            results["forecast_grid"]["stride_source"] = grid_src

            df = compute_residuals(actual, predicted, datetime_index)
            entries.append((model, df, results))
            yaml_paths[model] = OUTPUT_DIR / f"{model}_{real_h}h_residual_analysis.yaml"

            print_residual_summary(results)
            total_analyzed += 1

        if not entries:
            continue

        plot_files = save_grouped_plots(
            [(model, df) for model, df, _ in entries],
            horizon=horizon,
            output_dir=OUTPUT_DIR / "plots",
        )

        for model, _, results in entries:
            results["plot_files"] = [str(p) for p in plot_files]
            dump_residual_yaml(results, yaml_paths[model])

        print(f"\n{'─' * 80}")
        print(f"  Gráficos agrupados — {horizon}h "
              f"({len(entries)} modelo(s) en común, {len(plot_files)} PNG):")
        for p in plot_files:
            print(f"    → {p}")
        for model, _, _ in entries:
            print(f"    YAML → {yaml_paths[model]}")

    print(f"\n{'=' * 80}")
    print(f"Análisis completado: {total_analyzed} modelo(s) x horizonte(s)")
    if total_skipped:
        print(f"OMITIDOS por CSV no analizable: {total_skipped}")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    main()
