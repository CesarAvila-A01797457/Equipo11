# MiBici Equipo 11

Proyecto de MLOps con estructura Cookiecutter Data Science para preparar y explorar
salidas diarias por estación. Se procesó el histórico de diciembre de 2014 a
septiembre de 2026: **142 CSV mensuales y un catálogo actual**.

La fase implementada comprende descarga, limpieza por bloques, EDA y preprocesamiento.
Los datos describen viajes realizados; no disponibilidad ni demanda insatisfecha.
No se entrena un modelo final en esta fase.

## Entorno

Validado localmente en Windows con **Python 3.13.1**. El proyecto admite Python
3.13 y 3.14; la ejecución documentada corresponde a 3.13. Desde CMD:

```bat
py -3.13 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python -m pip check
```

Para reproducir las versiones exactas del entorno Windows utilizado, instalar
`requirements-lock-windows-py313.txt` cuando esté disponible y después el paquete
editable con `python -m pip install --no-deps -e .`. Ese archivo fija un entorno
Windows/Python 3.13; no se presenta como lock universal para otros sistemas.

## 1 Requerimientos y alcance

Se busca apoyar la planeación con futuras estimaciones de salidas por estación y
día. El ML Canvas y la justificación de valor deben acompañar la presentación.
Este repositorio conserva los datos observados y documenta los supuestos de sus
transformaciones. Los estación-días ausentes no se completan automáticamente con
cero: hace falta información histórica de operación para justificarlo.

## 2 Descarga y preparación

Fuente: [Datos abiertos de MiBici](https://mibici.net/en/open-data/).
La lista completa está en `references/mibici_urls_2014_2026.txt`.

```bat
python -m MiBici.dataset
python -m MiBici.prepare_data_streaming --chunk-size 25000
```

El descargador conserva archivos existentes salvo que se indique `--force` y
registra URL, tamaño, fecha, estado y SHA-256 en `data/raw/mibici/manifest.csv`.
La preparación por bloques lee solo Viaje_Id, Inicio_del_viaje y Origen_Id, valida
sus valores y conserva el primer viaje válido por mes e ID. SQLite temporal en
disco permite detectar duplicados entre bloques sin cargar el histórico en RAM.

El resultado y reporte quedan en `data/processed/historico/`. Se comprobaron
39,078,984 viajes leídos, 758 duplicados excluidos, 39,078,226 viajes retenidos y
1,161,039 filas agregadas. Los archivos originales no se modifican.

`MiBici.prepare_data` conserva la implementación anterior, que carga todos los
viajes en memoria. Para el histórico y equipos con RAM limitada se utiliza la
versión por bloques. La lista `references/mibici_urls_fase1.txt` se conserva como
referencia de la prueba inicial con cinco meses.

## 3 EDA y preprocesamiento

La preparación imprime un run_id. Sustituir `ID_DE_TU_PREPARACION` por el valor de
la ejecución propia al lanzar:

```bat
python -m MiBici.eda --source-run-id ID_DE_TU_PREPARACION
python -m MiBici.features --source-run-id ID_DE_TU_PREPARACION
```

El EDA comprueba la conciliación con el reporte de preparación y genera ocho
figuras, ocho tablas y conclusiones en `reports/eda_historico/`. No modifica el CSV.
Ver [informe](reports/eda_historico/EDA_MiBici.md) y
[metodología](references/EDA_METODOLOGIA.md).

El notebook `notebooks/01_eda_mibici_historico.ipynb` llama al módulo y presenta
resultados. Su exportación `reports/eda_historico/EDA_MiBici_Historico.html` permite
leerlos sin ejecutar código. Para abrir y ejecutar el notebook opcionalmente:

```bat
python -m pip install jupyterlab ipykernel
python -m jupyterlab notebooks/01_eda_mibici_historico.ipynb
```

El preprocesamiento genera calendario y utiliza one-hot para estación, día de
semana y mes; estandariza el año con StandardScaler. Salidas es el objetivo, sin
escalar y excluido de los predictores. Se ajusta exclusivamente con entrenamiento.

| Conjunto | Periodo | Filas |
| --- | --- | ---: |
| Entrenamiento | 2014-12-01 a 2025-12-31 | 1,052,704 |
| Validación | 2026-01-01 a 2026-06-30 | 66,724 |
| Prueba | 2026-07-01 a 2026-09-30 | 41,611 |

Los cortes pueden cambiarse con `--train-end` y `--validation-end`. Se conservan
todas las filas y salidas. Las 406 columnas se guardan en matrices dispersas,
junto con etiquetas, CSV por split, orden de columnas, transformador y manifiesto
en `data/processed/preprocesado/`.

Se reportan 55 estaciones de validación y 99 de prueba no vistas en entrenamiento.
Sus categorías se toleran con `handle_unknown=ignore`; esto no garantiza precisión
predictiva para estaciones nuevas. No se imputan ceros ni añaden rezagos en esta
versión. Ver [transformaciones, validaciones y límites](references/PREPROCESAMIENTO.md).

## 4 Versiones y trazabilidad en MLflow

Experimento de las etapas históricas: **mibici-fase1-historico**.

| Ejecución | Resultado registrado |
| --- | --- |
| prepare-mibici-demand-streaming | Agregado limpio, reporte, catálogo, manifiesto, código y huellas de fuentes. |
| eda-historico-mibici | EDA del agregado, tablas, figuras, métricas y código; conserva los valores del dataset. |
| preprocesamiento-calendario-mibici | Versión calendar_onehot_v1, tres entradas, matrices, etiquetas, transformador, reporte y huellas. |

Los runs de EDA y preprocesamiento guardan `preparation_run_id` y comparan la huella
del CSV con `output_sha256` de la preparación indicada. Los artefactos guardan
copias recuperables, además de los metadatos registrados con `log_input`.

```bat
python -m mlflow ui --backend-store-uri sqlite:///data/mlflow.db --host 127.0.0.1 --port 5000
```

Abrir http://127.0.0.1:5000 desde la raíz del proyecto. La base SQLite y los
artefactos están en `data/mlflow.db` y `data/mlartifacts/`, excluidos de Git.
**Un push no publica el almacén MLflow.** Otro integrante ejecuta el pipeline para
crear sus runs e IDs propios. El almacén usa rutas locales de artefactos; moverlo
requiere migración o una ejecución nueva, no solo copiar la base SQLite.

La [evidencia de ejecución local](reports/EJECUCION_LOCAL.md) documenta los IDs y
resultados confirmados del 24 de octubre de 2026. Para reproducir archivos idénticos
se deben contrastar las huellas y usar las mismas fuentes y versiones de software.

## 5 Organización y verificación

- `MiBici/dataset.py`: descarga y manifiesto.
- `MiBici/prepare_data_streaming.py`: validación, deduplicación y agregado por bloques.
- `MiBici/eda.py`: análisis, gráficos, tablas y reporte; `plots.py` delega en este módulo.
- `MiBici/features.py`: calendario, splits temporales y transformador.
- `tests/`: validaciones, conservación, ausencia de fuga y coherencia de artefactos.
- `references/`: fuentes y documentación metodológica.
- `reports/`: evidencia agregada y resultados para la presentación.

```bat
python -m unittest discover -s tests -v
```

La suite actual contiene nueve pruebas. No depende de descargar el histórico:
utiliza fixtures pequeños y comprueba errores, fechas ausentes sin imputación,
correlaciones indefinidas, conservación del objetivo, ajuste solo con train,
estaciones nuevas y alineación de matrices y etiquetas.

El EDA de todo el periodo es retrospectivo. Al modelar se deben fijar decisiones
sin optimizar sobre prueba y usar una evaluación temporal adecuada. No se presentan
métricas de un modelo final ni se afirma operación para cualquier fecha futura.

## Colaboración y entrega

Trabajar en ramas y registrar contribuciones reales de cada integrante. Antes de
publicar, revisar `git status` y los archivos seleccionados. Los CSV originales,
entornos y base de MLflow no se suben a Git.

La entrega académica debe incluir el PDF **Fase 1_Equipo11.pdf**, el ML Canvas,
las cinco etapas, evidencias de MLflow, enlace al repositorio con actividad de todos
y explicación de roles e interacción. La presentación se mantiene como entregable
separado del código.
