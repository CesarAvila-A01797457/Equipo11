# MiBici Equipo 11

Proyecto de MLOps con estructura Cookiecutter Data Science para preparar y explorar
salidas diarias por estación. Se procesó el histórico de diciembre de 2014 a
septiembre de 2026: **142 CSV mensuales y un catálogo actual**.

La fase implementada comprende descarga, limpieza por bloques, EDA y preprocesamiento.
Los datos describen viajes realizados; no disponibilidad ni demanda insatisfecha.
La Fase 2 añade entrenamiento, evaluación temporal y registro del modelo.

## Entorno

El artefacto de modelo usado como referencia para Fase 3 se generó en macOS con
**Python 3.14.7**. Las dependencias directas están fijadas en `requirements.txt` y
el lock completo del entorno está en `requirements-lock-macos-py314.txt`.
`requirements-lock-windows-py313.txt` corresponde a otro entorno: Windows/Python 3.13.

Para preparar un entorno limpio en macOS:

```bash
python3.14 -m venv .venv-repro
source .venv-repro/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

El entorno `.venv-repro` queda excluido de Git.

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
- `tests/`: limpieza, EDA, preprocesamiento, métricas e integración hasta la evaluación.
- `references/`: fuentes y documentación metodológica.
- `reports/`: evidencia agregada y resultados para la presentación.

Instala las dependencias del proyecto y ejecuta todas las pruebas desde la raíz:

```bash
pytest -q
```

La suite combina las pruebas existentes de limpieza, EDA y preprocesamiento con
pruebas del cálculo de métricas y una integración del flujo CSV → preparación de
variables → entrenamiento → predicción → evaluación. Utiliza archivos pequeños y
no descarga el histórico de MiBici.

El EDA de todo el periodo es retrospectivo. Al modelar se deben fijar decisiones
sin optimizar sobre prueba y usar una evaluación temporal adecuada. No se presentan
métricas de un modelo final ni se afirma operación para cualquier fecha futura.

## Fase 2: reproducción del entrenamiento

Ejecuta los comandos desde la raíz del repositorio en CMD con el entorno de la
sección **Entorno** activado. Para replicar el entorno de Windows, usa
`requirements-lock-windows-py313.txt` según las instrucciones de esa sección.

### 1. Generar el histórico filtrado

```bat
python -m MiBici.dataset
python -m MiBici.prepare_data_streaming --chunk-size 25000 --output-dir data/processed/historico_filtrado --experiment-name mibici-fase1-historico-filtrado
```

El entrenamiento usa el CSV agregado y su reporte de calidad de
`data/processed/historico_filtrado/`. Si ya los generaste con este flujo, puedes
reutilizarlos. Consulta el [reporte de preparación](reports/EJECUCION_FILTRADA.md)
para comprobar los conteos. El pipeline vuelve a ajustar el preprocesador con
entrenamiento y recibe el CSV agregado directamente.

### 2. Comparar configuraciones en validación

```bat
python -m MiBici.modeling.train --baseline --experiment mibici-fase2-modelado
for %A in (0.01 0.1 1.0) do python -m MiBici.modeling.train --poisson --alpha %A --experiment mibici-fase2-modelado
for %C in (1 2 3) do python -m MiBici.modeling.train --random-forest --forest-config %C --experiment mibici-fase2-modelado
```

El modelo estima salidas por estación y día. La referencia predice la media del
entrenamiento. Poisson usa estación, día de la semana, mes y año para estimar un
conteo esperado. El pipeline ajusta la codificación y la escala exclusivamente
con entrenamiento.

Se comparan tres configuraciones de Random Forest, con criterio Poisson y muestras
parciales para controlar el costo de entrenamiento:

| Configuración RF | Árboles | Profundidad máxima | Mínimo por hoja | `max_features` | Muestra por árbol |
|---|---:|---:|---:|---:|---:|
| 1 | 50 | 12 | 20 | 0.3 | 10 % |
| 2 | 75 | 16 | 20 | 0.3 | 15 % |
| 3 | 100 | 20 | 10 | 0.3 | 20 % |

Todas usan `random_state=42`. Cada árbol usa la fracción indicada del conjunto de
entrenamiento.

Poisson se compara variando su hiperparámetro de regularización `alpha`; las
demás opciones se mantienen iguales:

| Configuración Poisson | `alpha` | `max_iter` |
|---|---:|---:|
| 1 | 0.01 | 500 |
| 2 | 0.1 | 500 |
| 3 | 1.0 | 500 |

| Partición | Fechas observadas | Filas |
|---|---|---:|
| Entrenamiento | 2014-12-01 a 2025-12-31 | 1,052,321 |
| Validación | 2026-01-01 a 2026-06-30 | 66,715 |
| Prueba | 2026-07-01 a 2026-09-30 | 41,600 |

### Comparación de modelos en validación

| Modelo | Configuración | MAE | Sesgo |
|---|---|---:|---:|
| Referencia: media histórica | — | 22.0698 | +2.6079 |
| Poisson | `alpha=0.01` | **12.0651** | +8.2518 |
| Poisson | `alpha=0.1` | 15.7296 | +8.3819 |
| Poisson | `alpha=1.0` | 22.0650 | +7.8112 |
| Random Forest | Configuración 1 | 18.6756 | +3.4304 |
| Random Forest | Configuración 2 | 17.8918 | +3.2510 |
| Random Forest | Configuración 3 | 17.2976 | +3.1154 |

Se elige Poisson con `alpha=0.01` para el paso 3 porque obtuvo el menor MAE en
validación. Random Forest configuración 3 fue el mejor de sus tres candidatos,
pero no superó a Poisson. El sesgo se conserva como métrica complementaria: todos
los modelos sobreestimaron en promedio y Poisson tuvo el mayor sesgo. Las corridas
y sus parámetros también quedan registrados en el experimento de MLflow.

### 3. Evaluar en prueba y registrar el pipeline

Random Forest se compara en validación en la sección anterior. Poisson con
`alpha=0.01` obtuvo el menor MAE, así que se conserva el flujo original para la
prueba: evaluar la referencia y Poisson, y registrar este último.

```bat
python -m MiBici.modeling.train --baseline --test --experiment mibici-fase2-modelado
python -m MiBici.modeling.train --poisson --alpha 0.01 --test --experiment mibici-fase2-modelado --register-model mibici-poisson
```

| Modelo | MAE en prueba | Sesgo en prueba |
|---|---:|---:|
| Media histórica | 23.0970 | +3.5962 |
| Poisson con alpha=0.01 | 13.6708 | +7.9757 |

Las métricas se expresan en salidas por estación y día. El sesgo es el promedio
de predicción menos valor observado. En prueba, Poisson redujo el MAE un 40.8 %
frente a la referencia y tendió a sobreestimar.

### 4. Comprobar MLflow y recargar el modelo

```bat
python -m mlflow ui --backend-store-uri sqlite:///data/mlflow.db --host 127.0.0.1 --port 5000
```

En [MLflow local](http://127.0.0.1:5000), abre el experimento
`mibici-fase2-modelado` y compara las corridas de validación entre sí y las de
prueba entre sí. Revisa `input_sha256`, los cortes y las métricas de cada corrida.
En el registro de modelos, abre `mibici-poisson` y localiza la versión vinculada
a la corrida de prueba que guardó el pipeline.

Desde otro CMD con el entorno activado, verifica la recarga con una fila de
ejemplo. Sustituye `1` por la versión que imprimió tu registro.

```bat
python -c "import mlflow, mlflow.sklearn, pandas as pd; mlflow.set_tracking_uri('sqlite:///data/mlflow.db'); model = mlflow.sklearn.load_model('models:/mibici-poisson/2'); row = pd.DataFrame({'estacion_id': ['1'], 'dia_semana': ['0'], 'mes': ['7'], 'anio': [2026]}); print('Salidas estimadas:', model.predict(row)[0])"
```

Cada clon genera sus propios IDs de corrida y versiones en su registro local.
La base SQLite y los artefactos de `mlruns/` se conservan localmente.

## Fase 3: servicio de predicción con FastAPI

El endpoint `POST /predict` recibe una fecha y una lista de IDs de estación. La API
deriva el día de la semana, el mes y el año, y carga el pipeline completo (preprocesador
y modelo) desde MLflow al iniciar. El URI registrado actualmente es
`models:/mibici-poisson/2`. La API no entrena ni registra modelos al recibir solicitudes.

Desde la raíz del proyecto y con el entorno virtual activo, instala las dependencias
actualizadas y ejecuta el servicio:

```bash
python -m pip install -r requirements.txt
python -m uvicorn MiBici.api:app --host 127.0.0.1 --port 8000
```

La documentación interactiva queda en `http://127.0.0.1:8000/docs`; el estado del
servicio se consulta en `http://127.0.0.1:8000/health`.

La API usa `models:/mibici-poisson/2` por defecto. Para seleccionar otra versión,
configura `MIBICI_MODEL_URI`, por ejemplo `models:/mibici-poisson/3`. Para apuntar a
otro almacén de MLflow, configura `MLFLOW_TRACKING_URI`. Si no se define, utiliza
`data/mlflow.db` del repositorio. El registro y los artefactos deben estar accesibles
desde el entorno donde se inicia el servicio.

### Schema de `POST /predict`

Solicitud JSON:

```json
{
  "fecha_objetivo": "2026-11-20",
  "estacion_ids": [1, 2, 3]
}
```

Respuesta JSON:

```json
{
  "fecha_objetivo": "2026-11-20",
  "predicciones": [
    {"estacion_id": 1, "salidas_estimadas": 12.5},
    {"estacion_id": 2, "salidas_estimadas": 8.2},
    {"estacion_id": 3, "salidas_estimadas": 4.7}
  ]
}
```

`fecha_objetivo` debe ser una fecha válida en formato ISO (`AAAA-MM-DD`). `estacion_ids`
debe contener IDs enteros positivos, únicos y no vacíos. FastAPI documenta los tipos y
las validaciones en OpenAPI/Swagger. Las estaciones no presentes durante el
entrenamiento se codifican como categorías desconocidas; su estimación puede ser menos
representativa.

### Verificación de reproducibilidad en un entorno limpio

La referencia corresponde a la versión 2 del modelo y a la ejecución de prueba
`a530ba141c7b4a52b5eec285359fad01`. El modelo está versionado en MLflow como
`models:/mibici-poisson/2`; `reports/fase3/reproducibility_reference.json` guarda
las métricas, la huella SHA-256 del dataset y tres predicciones de referencia.

El repositorio contiene punteros DVC para el snapshot procesado, su reporte de calidad
y una copia portable del artefacto del modelo. Los contenidos se recuperan con `dvc
pull`; la URI y el run originales de MLflow se conservan como referencia, pero la API
puede cargar el artefacto local sin conectarse a la base MLflow de la Mac.

#### Sincronizar datos y modelo con DVC

DVC mantiene en Git los archivos `.dvc` y guarda los contenidos en un remoto. Para este
proyecto se usa una carpeta dedicada dentro de OneDrive, sincronizada en ambos equipos.
Cada equipo configura su propia ruta local en `.dvc/config.local`, que no se sube a
Git. OneDrive sincroniza el contenido de esa carpeta; espera a que termine la
sincronización antes de ejecutar `dvc pull` en el otro equipo. Evita hacer `dvc push`
desde los dos equipos al mismo tiempo.

OneDrive ya está configurado en la Mac y los archivos se subieron con `dvc push` a
`MiBici-DVC`. La Cloud PC podrá recuperarlos cuando OneDrive haya terminado de
sincronizar esa carpeta y se configure allí su ruta local. La ruta de OneDrive no se
comparte en Git; cada equipo la guarda en `.dvc/config.local`.

En la Mac, instala DVC en un entorno separado para no cambiar las dependencias del
modelo y configura la ruta de la carpeta sincronizada:

```bash
python3.14 -m venv .venv-dvc
source .venv-dvc/bin/activate
python -m pip install -r requirements-dvc.txt
dvc remote add --local -d onedrive "/ruta/local/de/OneDrive/MiBici-DVC"
dvc push
```

Los datos versionados son `data/processed/historico_filtrado/mibici_salidas_estacion_fecha.csv`
y `mibici_quality_report.json`. El artefacto está en `models/mibici-poisson-v2/`.
Sus punteros son `data/processed/historico_filtrado/*.dvc` y
`models/mibici-poisson-v2.dvc`.

En la Cloud PC Windows, después de clonar la misma rama y de tener OneDrive instalado
y sincronizado, configura la ruta local equivalente y recupera los archivos:

```powershell
python -m venv .venv-dvc
.\.venv-dvc\Scripts\Activate.ps1
python -m pip install -r requirements-dvc.txt
dvc remote add --local -d onedrive "C:\Users\<usuario>\OneDrive\MiBici-DVC"
dvc pull
```

La ubicación de OneDrive varía entre cuentas; reemplaza las rutas de ejemplo por la
carpeta real en cada equipo. DVC trata esa carpeta sincronizada como un remoto del
sistema de archivos, no como una conexión directa a la API de OneDrive. [DVC admite
remotos de sistema de archivos](https://doc.dvc.org/user-guide/data-management/remote-storage).

#### Ejecutar la comprobación

En la Cloud PC, crea el entorno de aplicación con Python y dependencias compatibles
con el artefacto. El modelo se guardó con Python 3.14.7, scikit-learn 1.9.1 y las
versiones que aparecen en `models/mibici-poisson-v2/requirements.txt`. Usa esas
versiones para cargar el artefacto; el lock de Windows/Python 3.13 corresponde a otro
entorno y no debe usarse para esta comprobación. Inicia la API con el modelo que DVC
recuperó:

```powershell
python -m venv .venv-repro
.\.venv-repro\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:MIBICI_MODEL_URI = "models/mibici-poisson-v2"
python -m uvicorn MiBici.api:app --host 127.0.0.1 --port 8001
```

En una segunda terminal, activa el mismo entorno y compara las métricas de prueba, las
predicciones directas y la respuesta del servicio con la URI original de MLflow:

```powershell
.\.venv-repro\Scripts\Activate.ps1
$env:MIBICI_MODEL_URI = "models/mibici-poisson-v2"
python -m MiBici.reproducibility `
  --model-uri "models/mibici-poisson-v2" `
  --reference-model-uri "models:/mibici-poisson/2" `
  --api-url http://127.0.0.1:8001/predict
```

El comando compara la huella SHA-256 del dataset, el número de filas, el MAE y el
sesgo en prueba, y las tres predicciones del ejemplo, con tolerancia absoluta de
`0.0001`. También reporta las diferencias de versiones entre entornos. Guarda el
resultado en `reports/fase3/reproducibility_clean_env.json`. No entrena ni crea
corridas de MLflow. El reporte puede versionarse en Git como evidencia.

La verificación carga desde DVC una copia del artefacto asociado a
`models:/mibici-poisson/2`; no necesita acceder al registro MLflow de la Mac. DVC
versiona también el snapshot exacto del dataset usado en la evaluación.

El Random Forest usa `RANDOM_SEED = 42` en `MiBici/config.py`; la regresión Poisson
utilizada como artefacto de referencia no tiene un componente aleatorio.

Este procedimiento usa una Cloud PC Windows como entorno separado. La reproducción
requiere recuperar los contenidos DVC antes de iniciar la API y el verificador.
