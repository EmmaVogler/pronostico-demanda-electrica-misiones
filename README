# Pronóstico de demanda eléctrica de Misiones

Proyecto en Python para optimizar y evaluar modelos de pronóstico de demanda eléctrica con **SARIMAX**, **LightGBM**, **LSTM** y **Chronos-2**. Incluye scripts de análisis y notebooks para explorar los datos y generar figuras.

## Datos

Los datos de demanda son de terceros y **no están incluidos** en este repositorio. Antes de utilizarlos, obténlos de la fuente autorizada y verifica que cuentas con permiso para hacerlo. Coloca los archivos en las rutas indicadas en [`config.yaml`](config.yaml), dentro de `data/raw` y `data/processed`.

Los JSON meteorológicos previos al inicio de la serie también se excluyen. Son necesarios únicamente si se vuelven a ejecutar los scripts de preparación de datos; consulta [`data/README.md`](data/README.md).

## Requisitos

- Windows
- Python 3.12
- Entorno compatible con la versión de PyTorch indicada en [`requirements.txt`](requirements.txt)
- Conexión a Internet para instalar dependencias y descargar Chronos-2 la primera vez

## Instalación

Abre PowerShell en la carpeta del proyecto y ejecuta:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Optimización y evaluación

Configura `forecast_horizon`, `stride` y `context_window` en la sección `backtesting` de `config.yaml`. La tesis evalúa dos escenarios:

| Escenario | `forecast_horizon` | `stride` | `context_window` |
| --- | ---: | ---: | ---: |
| Pronóstico a 3 horas | `3` | `3` | `168` |
| Pronóstico a 24 horas (day-ahead) | `24` | `24` | `168` |

Ejecuta la optimización y luego la evaluación para cada modelo y escenario. Cambia la configuración entre escenarios y vuelve a ejecutar los comandos:

```powershell
python src\optimization\sarimax_optuna.py
python src\evaluation\sarimax_evaluation.py

python src\optimization\lightgbm_optuna.py
python src\evaluation\lightgbm_evaluation.py

python src\optimization\lstm_optuna.py
python src\evaluation\lstm_evaluation.py

python src\optimization\chronos2_optuna.py
python src\evaluation\chronos2_evaluation.py
```

`stride` es el avance entre ventanas de pronóstico. Al igualarlo al horizonte, las ventanas son consecutivas y no se superponen. `context_window` indica cuántas horas previas usa el modelo; `168` corresponde a una semana.

En el flujo actual, conserva `retrain: false` y `last_points_only: false`. Los scripts fijan estos valores en las llamadas de backtesting; cambiarlos en `config.yaml` no altera ese comportamiento. `last_points_only: false` conserva las predicciones de todo el horizonte para su evaluación.

Los resultados se guardan en `output/`. La optimización puede tardar y actualizar los mejores parámetros en `config.yaml`.

## Análisis de residuos

Después de completar las evaluaciones, ejecuta:

```powershell
python src\analysis\run_residual_analysis.py
```

## Notebooks

Los notebooks son análisis complementarios. Ábrelos desde VS Code o Jupyter con el kernel del entorno del proyecto: **`.venv` (Python 3.12)**. Elige el intérprete `.\.venv\Scripts\python.exe`; `ipykernel` se instala junto con las dependencias.

| Notebook | Contenido |
| --- | --- |
| [Análisis del dataset](notebooks/%5Bstage-1%5Danalisis-dataset.ipynb) | Exploración del dataset, patrones temporales y diagnósticos de estacionariedad. |
| [Análisis de trials](notebooks/%5Bstage-2%5Danalisis-trials.ipynb) | Análisis y comparación de los resultados de optimización. |
| [Figuras del informe](notebooks/%5Bstage-fina%5Dfiguras-informe.ipynb) | Generación de figuras del informe a partir de datos y resultados. |

El notebook de etapa 1 requiere el archivo de `data/raw`. Los de etapas 2 y final requieren los resultados correspondientes de `output/optimization` y `output/evaluation`.
