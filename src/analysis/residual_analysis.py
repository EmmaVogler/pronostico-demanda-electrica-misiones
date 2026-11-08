"""
Módulo de análisis de residuos para modelos de forecasting.

Funciones reutilizables para diagnosticar la calidad de las predicciones:
- Normalidad (Shapiro-Wilk)
- Autocorrelación (ACF / PACF)
- Homocedasticidad (Spearman / Goldfeld-Quandt)
- Sesgos sistemáticos (MAE por hora, mes, día, estación)
- Gráficos de diagnóstico (agrupados: un PNG por horizonte × tipo de test)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import shapiro, spearmanr, norm, skew, kurtosis
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.diagnostic import het_goldfeldquandt
from ruamel.yaml import YAML

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

MIN_OBSERVATIONS = 50

def compute_residuals(actual, predicted, datetime_index):
    """Calcula residuos y devuelve DataFrame limpio.

    Parameters
    ----------
    actual : array-like
        Valores reales del test set.
    predicted : array-like
        Valores predichos por el modelo.
    datetime_index : array-like
        Datetimes correspondientes a cada observación.

    Returns
    -------
    pd.DataFrame con columnas: datetime, actual, predicted, residual
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    datetime_index = pd.DatetimeIndex(datetime_index)

    if not (len(datetime_index) == len(actual) == len(predicted)):
        raise ValueError(
            f"Desalineación de longitudes: datetime={len(datetime_index)}, "
            f"actual={len(actual)}, predicted={len(predicted)}. Deben ser "
            f"iguales y corresponder 1:1 en el mismo orden."
        )

    df = pd.DataFrame({
        "datetime": datetime_index,
        "actual": actual,
        "predicted": predicted,
    })

    df = df.sort_values("datetime", kind="stable").reset_index(drop=True)

    if df["datetime"].duplicated().any():
        n_dup = int(df["datetime"].duplicated().sum())
        raise ValueError(
            f"{n_dup} timestamps duplicados en datetime_index. Las ventanas "
            f"solapadas (horizon < stride) generan timestamps repetidos; "
            f"deduplicar antes de analizar residuos."
        )

    df["residual"] = df["actual"] - df["predicted"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["residual"])
    return df

def validate_forecast_grid(df, horizon=None, stride=None):
    """Valida que los timestamps correspondan a una grilla de ventanas
    ``historical_forecasts(forecast_horizon=horizon, stride=stride)``.

    Sin esto no se detectan los CSVs con timestamps fabricados: cuando
    ``stride > horizon`` las ventanas dejan huecos reales entre sí, así que la
    serie de predicciones NO es contigua. Un CSV generado con
    ``time_index[-n:]`` produce en cambio un bloque contiguo de 1h, que es
    estructuralmente válido (sin duplicados, longitudes correctas) pero
    semánticamente falso: cada etiqueta de tiempo queda desfasada respecto de
    la predicción que realmente le corresponde.

    relaciones esperadas entre deltas consecutivos (en horas):
        horizon <  stride -> dentro de ventana 1h, entre ventanas stride-horizon+1
        horizon == stride -> todos 1h (ventanas contiguas)
        horizon >  stride -> ventanas solapadas -> timestamps duplicados
                            (detectado antes, en compute_residuals)

    Raises
    ------
    ValueError si el patrón de deltas contradice la grilla esperada.
    """
    n = len(df)
    if horizon is None or stride is None or horizon <= 0 or stride <= 0:
        return {"forecast_grid_validated": False,
                "reason": "horizon/stride no provistos: grilla no verificada"}

    deltas = df["datetime"].diff().dt.total_seconds().div(3600).dropna()
    if deltas.empty:
        return {"forecast_grid_validated": False, "reason": "n<2 timestamps"}

    if horizon > stride:

        raise ValueError(
            f"horizon({horizon}) > stride({stride}) implica ventanas solapadas, "
            f"que deben producir timestamps duplicados. No se detectaron: el "
            f"CSV fue construido de forma incompatible con la grilla de forecast."
        )

    gap = stride - horizon + 1
    if gap == 1:
        if not (deltas == 1).all():
            raise ValueError(
                f"horizon({horizon}) == stride({stride}) exige timestamps "
                f"contiguos de 1h, pero se observaron deltas "
                f"{sorted(set(deltas.unique()))[:5]}. Grilla inconsistente."
            )
        n_windows = n // horizon
    else:
        if (deltas == 1).all():
            raise ValueError(
                f"horizon({horizon}) < stride({stride}) exige huecos de "
                f"{gap}h entre ventanas, pero TODOS los deltas son de 1h: los "
                f"timestamps son un bloque contiguo fabricated y no la grilla "
                f"real de ventanas. Regenere el CSV desde el evaluation script "
                f"(los timestamps deben salir de actual.time_index, no de "
                f"time_index[-n:])."
            )
        fuera_de_grilla = deltas[(deltas != 1) & (deltas != gap)]
        if len(fuera_de_grilla) > 0:
            raise ValueError(
                f"Se esperaban deltas de 1h (dentro de ventana) o {gap}h "
                f"(entre ventanas), pero hay {len(fuera_de_grilla)} deltas "
                f"fuera de grilla: {sorted(set(fuera_de_grilla.unique()))[:5]}."
            )
        n_windows = int((deltas != 1).sum()) + 1

    return {
        "forecast_grid_validated": True,
        "horizon": int(horizon),
        "stride": int(stride),
        "gap_hours": int(gap),
        "n_windows": int(n_windows),
        "span_hours": int((df['datetime'].iloc[-1] - df['datetime'].iloc[0]).total_seconds() // 3600),
    }

def _classify(value, thresholds, labels):
    """Clasifica un valor no negativo según umbrales crecientes."""
    for threshold, label in zip(thresholds, labels):
        if value < threshold:
            return label
    return labels[-1]

def residual_distribution_metrics(residuals):
    """Tamaños de efecto de la forma de la distribución de residuos.

    Existe porque el p-valor de Shapiro-Wilk no discrimina a esta escala. Con
    n ~ 4000 la potencia del test es tan alta que rechaza H0 ante desviaciones
    irrelevantes para forecasting. Referencia medida a n = 3936:

        distribucion                sesgo   kurtosis   p(Shapiro)
        normal                       0.00      -0.02        4.1e-01
        t-Student df=30              0.01       0.27        2.0e-02
        t-Student df=10              0.07       0.50        1.3e-07
        t-Student df=5               0.31       5.76        3.2e-29
        mezcla 97/3 con outliers     0.15      26.21        2.9e-60

    Una t-Student df=30 es visualmente normal y NO rechaza. Reach p ~ 1e-50
    exige kurtosis ~20, equivalente a contaminacion con ~3% de outliers
    severos. Por eso se reporta la magnitud, no solo la significancia.
    """
    r = np.asarray(residuals, dtype=float)
    n = len(r)
    if n < MIN_OBSERVATIONS or np.isclose(r.std(), 0):
        return {"n": int(n), "note": f"n={n} < {MIN_OBSERVATIONS}: no aplicable"}

    rmse = float(np.sqrt(np.mean(r ** 2)))
    sk = float(skew(r))
    ku = float(kurtosis(r, fisher=True))
    z = r / rmse

    k_worst = max(1, int(0.01 * n))
    sq = np.sort(r ** 2)
    share_worst = float(np.sum(sq[-k_worst:]) / np.sum(sq)) if np.sum(sq) > 0 else 0.0

    return {
        "n": int(n),
        "rmse": rmse,
        "skewness": round(sk, 4),
        "excess_kurtosis": round(ku, 4),
        "skewness_magnitude": _classify(
            abs(sk), [0.2, 0.5, 1.0],
            ["despreciable", "leve", "moderado", "severo"]),
        "kurtosis_magnitude": _classify(
            abs(ku), [0.5, 3.0, 7.0],
            ["despreciable", "leve", "moderado", "severo"]),
        "pct_beyond_2sigma": round(float(100 * np.mean(np.abs(z) > 2)), 3),
        "pct_beyond_3sigma": round(float(100 * np.mean(np.abs(z) > 3)), 3),
        "pct_beyond_2sigma_normal_ref": 4.55,
        "pct_beyond_3sigma_normal_ref": 0.27,
        "max_abs_zscore": round(float(np.max(np.abs(z))), 3),
        "worst_1pct_share_of_mse": round(100 * share_worst, 2),
        "asymmetry_low_tail": round(float(np.mean(r < -2 * rmse)), 5),
        "asymmetry_high_tail": round(float(np.mean(r > 2 * rmse)), 5),
        "interpretation": (
            f"kurtosis {ku:.1f} equivale a contaminacion con ~"
            f"{max(0.0, (abs(ku) - 3.0) / 6.0) * 3:.1f}% de outliers severos; "
            f"el peor 1% de las horas aporta {100 * share_worst:.1f}% del "
            f"error cuadratico"
        ),
    }

def test_normality(residuals, max_n=5000, seed=42):
    """Test de Shapiro-Wilk sobre los residuos.

    H0: los residuos se distribuyen normalmente.
    Si p > 0.05 → no se rechaza H0 (normalidad aceptada).

    Si n > max_n se submuestrea de forma reproducible (seed fijo) para no
    exceder el límite de Shapiro-Wilk y para que el resultado sea
    determinístico entre corridas.
    """
    if len(residuals) > max_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(residuals), max_n, replace=False)
        sample = residuals[idx]
    else:
        sample = residuals
    sw_stat, sw_pval = shapiro(sample)
    eff = residual_distribution_metrics(residuals)

    return {
        "shapiro_stat": float(sw_stat),
        "shapiro_pval": float(sw_pval),
        "shapiro_normal": bool(sw_pval > 0.05),
        "effect_sizes": eff,
        "conclusion": (
            "rechaza H0 pero desviacion NO material (kurtosis "
            f"{eff.get('excess_kurtosis', float('nan')):.2f} <= 3): "
            "rechazo por potencia, no por severidad"
            if eff.get("kurtosis_magnitude") in ("despreciable", "leve")
            else "desviacion material: Kolmogorov-Smirnov asume normalidad "
                 "pero no afecta la unbiasedidad de MAE/RMSE"
        ),
    }

def test_autocorrelation(residuals, max_lag=168, max_pacf_lag=48, min_n=8):
    """Calcula ACF y PACF de los residuos en lags clave.

    Retorna valores numéricos en lags 1, 24, 168 y los arreglos completos
    para graficar.

    El número de lags se limita a ``n // 2 - 1`` porque statsmodels no puede
    calcular PACF con ``nlags > n // 2`` (lanza ValueError), y porque más allá
    de la mitad de la muestra el estimador de autocorrelación es poco
    confiable.

    Se reporta ``acf_max_lag_computed``: si es menor que 168, el lag 168
    (estacionalidad semanal) NO pudo calcularse por falta de muestra. Con
    n=288 el techo es 143, así que se necesita n >= 338 para medir la semana.
    """
    n = len(residuals)

    if n < min_n:
        return {
            "acf_lag_1": None, "acf_lag_24": None, "acf_lag_168": None,
            "pacf_lag_1": None, "pacf_lag_24": None,
            "acf_max_lag_computed": 0,
            "acf1_variance_explained_pct": None,
            "acf_values": [], "pacf_values": [],
            "autocorrelation_note": f"n={n} < {min_n}: test no aplicable",
        }

    acf_lags = safe_max_lag(n, max_lag)
    acf_vals = acf(residuals, nlags=acf_lags)

    pacf_lags = safe_max_lag(n, max_pacf_lag)
    pacf_vals = pacf(residuals, nlags=pacf_lags)

    def _safe(arr, idx):
        return float(arr[idx]) if idx < len(arr) else None

    acf1 = _safe(acf_vals, 1)

    lb_k = min(24, acf_lags)
    lb = acorr_ljungbox(residuals, lags=[lb_k])
    lb_stat = float(np.atleast_1d(lb["lb_stat"])[0])
    lb_pval = float(np.atleast_1d(lb["lb_pvalue"])[0])

    return {
        "acf_lag_1": acf1,
        "acf_lag_24": _safe(acf_vals, 24),
        "acf_lag_168": _safe(acf_vals, 168),
        "pacf_lag_1": _safe(pacf_vals, 1),
        "pacf_lag_24": _safe(pacf_vals, 24),
        "acf_max_lag_computed": int(acf_lags),
        "acf1_variance_explained_pct": (
            round(100 * acf1 ** 2, 2) if acf1 is not None else None),
        "acf1_magnitude": _classify(
            abs(acf1), [0.1, 0.3, 0.5],
            ["despreciable", "leve", "moderada", "SEVERA"]) if acf1 is not None else None,
        "ljung_box_stat": lb_stat,
        "ljung_box_pval": lb_pval,
        "ljung_box_lags": int(lb_k),
        "acf_values": [float(v) for v in acf_vals.tolist()],
        "pacf_values": [float(v) for v in pacf_vals.tolist()],
    }

def test_heteroscedasticity(residuals, predicted, min_n=20):
    """Tests de homocedasticidad: Spearman y Goldfeld-Quandt.

    Spearman: correlación entre |residuos| y predichos.
    Si p < 0.05 → heterocedasticidad significativa.

    Goldfeld-Quandt: test de varianza constante. Se reportan DOS versiones
    porque miden cosas distintas:

    - ``*_by_prediction``: ordena los residuos por el valor predicho (idx=0) y
      parte la muestra en dos mitades. Responde: ¿la varianza de los residuos
      crece con el nivel de predicción? (heterocedasticidad por escala)
    - ``*_by_time``: NO ordena (idx=None), parte el test set en el orden
      temporal. Responde: ¿la varianza de los residuos cambió a lo largo del
      período evaluado? (cambio de régimen / no estacionariedad)

    Ojo: en el caso *by_time* el nombre "homocedástico" es engañoso, no es un
    test de homocedasticidad sino de estabilidad temporal de la varianza.
    """
    residuals = np.asarray(residuals, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    abs_resid = np.abs(residuals)
    corr, pval = spearmanr(predicted, abs_resid)

    out = {
        "spearman_corr": float(corr),
        "spearman_pval": float(pval),
        "heteroscedastic": bool(pval < 0.05),
    }

    if len(residuals) < min_n:
        out.update({
            "goldfeld_quandt_by_prediction_stat": None,
            "goldfeld_quandt_by_prediction_pval": None,
            "goldfeld_quandt_by_prediction_homoscedastic": None,
            "goldfeld_quandt_by_time_stat": None,
            "goldfeld_quandt_by_time_pval": None,
            "goldfeld_quandt_by_time_stable_variance": None,
            "goldfeld_quandt_note": f"n={len(residuals)} < {min_n}: test no aplicable",
        })
        return out

    x = predicted.reshape(-1, 1)
    gq_pred_stat, gq_pred_pval, _ = het_goldfeldquandt(residuals, x, idx=0)
    gq_time_stat, gq_time_pval, _ = het_goldfeldquandt(residuals, x, idx=None)

    out.update({
        "goldfeld_quandt_by_prediction_stat": float(gq_pred_stat),
        "goldfeld_quandt_by_prediction_pval": float(gq_pred_pval),
        "goldfeld_quandt_by_prediction_homoscedastic": bool(gq_pred_pval > 0.05),
        "goldfeld_quandt_by_time_stat": float(gq_time_stat),
        "goldfeld_quandt_by_time_pval": float(gq_time_pval),
        "goldfeld_quandt_by_time_stable_variance": bool(gq_time_pval > 0.05),
    })
    return out

def analyze_systematic_biases(df):
    """MAE de residuos por hora, mes, día de semana y estación."""
    df = df.copy()
    df["hour"] = df["datetime"].dt.hour
    df["month"] = df["datetime"].dt.month
    df["dow"] = df["datetime"].dt.dayofweek
    df["season"] = df["month"].map({
        12: "verano", 1: "verano", 2: "verano",
        3: "otoño", 4: "otoño", 5: "otoño",
        6: "invierno", 7: "invierno", 8: "invierno",
        9: "primavera", 10: "primavera", 11: "primavera",
    })

    def _mae_by(grouped):
        return {name: float(np.mean(np.abs(group.values))) for name, group in grouped}

    return {
        "mean_bias": float(df["residual"].mean()),
        "mae_by_hour": _mae_by(df.groupby("hour")["residual"]),
        "mae_by_month": _mae_by(df.groupby("month")["residual"]),
        "mae_by_dow": _mae_by(df.groupby("dow")["residual"]),
        "mae_by_season": _mae_by(df.groupby("season")["residual"]),
    }

def safe_max_lag(n, desired):
    """Máximo número de lags utilizable para ACF/PACF dado el tamaño de muestra.

    statsmodels lanza ValueError en pacf() si ``nlags > n // 2``, y estimate
    -por encima de la mitad de la muestra- es poco confiable. Además
    ``plot_acf`` rechaza lags negativos, lo que rompía con n < 3.
    """
    return max(1, min(desired, n // 2 - 1))

def _draw_serie(ax, df):
    """Serie temporal de residuos con banda de ±2σ."""
    residuals = df["residual"].values
    sigma = np.std(residuals)
    ax.plot(df["datetime"], residuals, linewidth=0.5, color="#2c7bb6")
    ax.axhline(0, color="red", linestyle="--", linewidth=0.8)
    ax.axhline(2 * sigma, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(-2 * sigma, color="gray", linestyle=":", alpha=0.5)
    ax.fill_between(df["datetime"], -2 * sigma, 2 * sigma, alpha=0.05, color="gray")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Residuo (MW)")
    ax.tick_params(axis="x", rotation=30)

def _draw_histogram(ax, df):
    """Histograma de residuos contra la normal teórica."""
    residuals = df["residual"].values
    sigma = np.std(residuals)
    ax.hist(residuals, bins=50, density=True, alpha=0.7, edgecolor="black",
            color="#abd9e9", label="Residuos")
    x = np.linspace(residuals.min(), residuals.max(), 200)
    ax.plot(x, norm.pdf(x, residuals.mean(), sigma), "r-", linewidth=2,
            label=f"Normal(μ={residuals.mean():.4f}, σ={sigma:.4f})")
    ax.set_xlabel("Residuo (MW)")
    ax.set_ylabel("Densidad")
    ax.legend(fontsize=8)

def _draw_qqplot(ax, df):
    """QQ plot de residuos contra la normal."""
    qqplot(df["residual"].values, line="45", fit=True, ax=ax,
           markersize=3, alpha=0.5)
    if ax.get_lines():
        ax.get_lines()[0].set_markerfacecolor("#2c7bb6")

def _draw_acf(ax, df):
    """ACF de los residuos."""
    residuals = df["residual"].values
    plot_acf(residuals, lags=safe_max_lag(len(residuals), 168), ax=ax, alpha=0.05,
             color="#2c7bb6", vlines_kwargs={"colors": "#2c7bb6"})
    ax.set_xlabel("Lag (horas)")

def _draw_pacf(ax, df):
    """PACF de los residuos."""
    residuals = df["residual"].values
    plot_pacf(residuals, lags=safe_max_lag(len(residuals), 48), ax=ax, alpha=0.05,
              color="#2c7bb6", vlines_kwargs={"colors": "#2c7bb6"})
    ax.set_xlabel("Lag (horas)")

def _draw_vs_predicted(ax, df):
    """Residuos contra predicciones + recta de tendencia."""
    residuals = df["residual"].values
    ax.scatter(df["predicted"], residuals, alpha=0.3, s=10, color="#2c7bb6")
    ax.axhline(0, color="red", linestyle="--", linewidth=0.8)
    z = np.polyfit(df["predicted"].values, residuals, 1)
    p = np.poly1d(z)
    x_line = np.linspace(df["predicted"].min(), df["predicted"].max(), 100)
    ax.plot(x_line, p(x_line), "r-", linewidth=2,
            label=f"Tendencia (pendiente={z[0]:.4f})")
    ax.legend(fontsize=8)
    ax.set_xlabel("Predicción (MW)")
    ax.set_ylabel("Residuo (MW)")

_PANEL_DRAWERS = {
    "serie": _draw_serie,
    "histogram": _draw_histogram,
    "qqplot": _draw_qqplot,
    "acf": _draw_acf,
    "pacf": _draw_pacf,
    "vs_predicted": _draw_vs_predicted,
}

PLOT_GROUPS = {
    "serie": {
        "title": "Serie temporal de residuos",
        "panels": [("serie", "Serie de residuos")],
        "figsize": (14, 3.0),
    },
    "normalidad": {
        "title": "Distribución de residuos (normalidad)",
        "panels": [("histogram", "Histograma + Normal"),
                   ("qqplot", "QQ plot vs Normal")],
        "figsize": (13, 3.6),
    },
    "autocorrelacion": {
        "title": "Autocorrelación de residuos (ACF / PACF)",
        "panels": [("acf", "ACF"), ("pacf", "PACF")],
        "figsize": (14, 3.0),
    },
    "homocedasticidad": {
        "title": "Homocedasticidad (residuos vs predichos)",
        "panels": [("vs_predicted", "Residuos vs predichos")],
        "figsize": (9, 4.5),
        "cols_por_fila": 2,
    },
}

def save_grouped_plots(entries, horizon, output_dir, dpi=150):
    """Genera los gráficos de diagnóstico AGRUPADOS por horizonte × tipo de test.

    Parameters
    ----------
    entries : list of (str, pd.DataFrame)
        Tuplas ``(model_name, df_residuos)`` de TODOS los modelos que
        corresponda comparar en el mismo horizonte. El DataFrame debe ser la
        salida de ``compute_residuals`` (columnas datetime/actual/predicted/
        residual).
    horizon : int
        Horizonte de predicción en horas.
    output_dir : str o Path
        Directorio de salida.

    Returns
    -------
    list de Path — un PNG por tipo de test; los modelos van uno por
    renglón, salvo que el grupo fije "cols_por_fila" (grilla 2x2).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = [(str(model), df) for model, df in entries]
    if not entries:
        return []

    files = []
    for group_key, group in PLOT_GROUPS.items():
        panels = group["panels"]
        por_fila = group.get("cols_por_fila", 1)
        n_panels = len(panels)
        n_rows = (len(entries) + por_fila - 1) // por_fila
        n_cols = n_panels * por_fila
        width, height_per_row = group["figsize"]

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(width, height_per_row * n_rows),
            squeeze=False,
        )

        for idx, (model_name, df) in enumerate(entries):
            row, base_col = divmod(idx, por_fila)
            base_col *= n_panels
            for panel, (panel_key, panel_label) in enumerate(panels):
                ax = axes[row][base_col + panel]
                _PANEL_DRAWERS[panel_key](ax, df)

                label = (panel_label if panel
                         else f"{model_name.upper()} — {panel_label}")
                ax.set_title(label, fontsize=10, loc="left")

        for k in range(len(entries) * n_panels, n_rows * n_cols):
            axes[k // n_cols][k % n_cols].axis("off")

        fig.suptitle(f"{horizon}h — {group['title']}", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))

        path = output_dir / f"{horizon}h_residuals_{group_key}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        files.append(path)

    return files

def generate_plots(df, model_name, horizon, output_dir):
    """Gráficos de diagnóstico de UN modelo (wrapper de ``save_grouped_plots``).

    Genera las 4 imágenes por horizonte × tipo de test con un solo renglón.
    La agrupación real (todos los modelos del mismo horizonte en la misma
    imagen) la arma el script ``run_residual_analysis.py`` con
    ``save_grouped_plots``.
    """
    return save_grouped_plots([(model_name, df)], horizon, output_dir)

def dump_residual_yaml(results, yaml_path):
    """Escribe los resultados del análisis en YAML.

    Está separada de ``run_residual_analysis`` para que el caller pueda armar
    primero los gráficos agrupados por horizonte (un PNG por tipo de test con
    TODOS los modelos) y recién después volcar cada YAML con ``plot_files``
    ya completo.
    """
    yaml_path = Path(yaml_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_rt = YAML()
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml_rt.dump(results, f)
    return yaml_path

def run_residual_analysis(actual, predicted, datetime_index,
                          model_name, horizon, output_dir, stride=None,
                          save_plots=True, save_yaml=True):
    """Ejecuta el análisis completo de residuos.

    Parameters
    ----------
    actual : array-like
        Valores reales del test set.
    predicted : array-like
        Valores predichos por el modelo.
    datetime_index : array-like
        Datetimes de cada observación del test set.
    model_name : str
        Nombre del modelo ("sarimax", "lightgbm", "lstm", "chronos2").
    horizon : int
        Horizonte de predicción en horas (24, 3, 6, 12).
    output_dir : str o Path
        Directorio de salida para YAML y gráficos.
    stride : int, opcional
        Stride de las ventanas de backtesting. Si se provee, se valida que los
        timestamps correspondan a la grilla real
        ``historical_forecasts(horizon, stride)`` y no a un bloque contiguo
        fabricado. Ver ``validate_forecast_grid``.
    save_plots : bool
        Si True genera los 4 PNG (uno por horizonte × tipo de test) con este
        modelo como único renglón. Ponélo en False cuando el caller vaya a
        armar los gráficos agrupados con ``save_grouped_plots`` para todos los
        modelos del mismo horizonte (es lo que hace el script CLI).
    save_yaml : bool
        Si False no escribe el YAML; el caller lo vuelca después de tener
        ``plot_files`` completo (ver ``dump_residual_yaml``).

    Returns
    -------
    dict con todos los resultados del análisis.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = compute_residuals(actual, predicted, datetime_index)

    if len(df) < MIN_OBSERVATIONS:
        raise ValueError(
            f"Solo {len(df)} observaciones válidas (mínimo requerido: "
            f"{MIN_OBSERVATIONS}). Los tests estadísticos (Shapiro, ACF/PACF, "
            f"Goldfeld-Quandt) y los lags 24/168 no son interpretables con "
            f"muestra tan chica. No se genera análisis."
        )

    grid = validate_forecast_grid(df, horizon=horizon, stride=stride)

    normality = test_normality(df["residual"].values)
    autocorr = test_autocorrelation(df["residual"].values)
    heterosc = test_heteroscedasticity(df["residual"].values, df["predicted"].values)
    biases = analyze_systematic_biases(df)
    dist_metrics = normality["effect_sizes"]

    residuals = df["residual"].values
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    residual_stats = {
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals)),
        "min": float(np.min(residuals)),
        "max": float(np.max(residuals)),
        "median": float(np.median(residuals)),
        "rmse": rmse,
    }

    plots_dir = output_dir / "plots"
    plot_files = (generate_plots(df, model_name, horizon, plots_dir)
                  if save_plots else [])

    results = {
        "model": model_name,
        "horizon": horizon,
        "stride": stride,
        "forecast_grid": grid,
        "n_observations": len(df),
        "residual_stats": residual_stats,
        "tamano_efecto": dist_metrics,
        "normalidad": normality,
        "autocorrelacion": autocorr,
        "homocedasticidad": heterosc,
        "sesgos_sistematicos": biases,
        "plot_files": [str(p) for p in plot_files],
    }

    if save_yaml:
        dump_residual_yaml(results,
                           output_dir / f"{model_name}_{horizon}h_residual_analysis.yaml")

    return results
