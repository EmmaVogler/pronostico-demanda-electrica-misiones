# Datos del proyecto

Los archivos CSV de entrada y los conjuntos procesados no se incluyen en este repositorio. Deben obtenerse de su fuente autorizada y colocarse en las rutas configuradas en `config.yaml`, dentro de `data/raw` y `data/processed`.

## Archivos JSON meteorológicos

Los scripts de preparación pueden usar archivos JSON con observaciones horarias de temperatura y humedad previas al comienzo del dataset de demanda. Se emplean como historial para calcular rezagos al inicio de la serie: LightGBM, LSTM y Chronos-2 usan observaciones del 20 al 31 de diciembre de 2022 para variables retrasadas (por ejemplo, 1 y 24 horas); SARIMAX usa las del 31 de diciembre para calcular el rezago de 24 horas de las primeras observaciones. Si se ejecutan esos scripts, los JSON correspondientes deben estar disponibles en `src/data`.

## Scripts de preparación

- `humidity_temperature_features_SMN.py`: obtiene observaciones meteorológicas horarias del SMN y las combina con los datos de demanda.
- `humidity_temperature_features_open-meteo.py`: completa valores faltantes de temperatura y humedad consultando Open-Meteo.
- `forecast_features_open-meteo.py`: incorpora variables meteorológicas de pronóstico de Open-Meteo al conjunto de datos.
- `order_dataset.py`: ordena cronológicamente el conjunto de datos con variables meteorológicas.
- `sarimax_features.py`: prepara variables temporales, meteorológicas y rezagos para SARIMAX.
- `lightbgm_features.py`: genera variables temporales, rezagos y componentes Fourier para LightGBM.
- `lstm_features.py`: prepara variables temporales y rezagos para LSTM.
- `chronos2_features.py`: prepara el conjunto de datos y las covariables para Chronos-2.
