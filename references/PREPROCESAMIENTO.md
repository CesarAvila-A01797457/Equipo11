# Preprocesamiento del histórico MiBici

Esta primera versión demuestra generación de variables de calendario, codificación
categórica, estandarización y conservación de datos con trazabilidad en MLflow.
No entrena un modelo ni resuelve la disponibilidad histórica de estaciones.

## Transformaciones

| Campo | Tratamiento | Justificación |
| --- | --- | --- |
| estacion_id | OneHotEncoder | Es una categoría, aunque el ID sea un número. |
| dia_semana | Derivado de fecha y codificado one-hot | Representa calendario conocido antes de la predicción. Lunes=0, domingo=6. |
| mes | Derivado de fecha y codificado one-hot | Representa diferencias estacionales sin imponer una distancia numérica entre meses. |
| anio | StandardScaler ajustado solo con train | Representa una tendencia temporal; media cero y escala unitaria en entrenamiento. |
| salidas | Objetivo entero sin escalar | Se conserva el conteo original; nunca se usa como predictor. |
| fecha | Identificador y criterio de separación | Se conserva en CSV para rastrear filas, pero no se entrega directamente al transformador. |

No se imputa: el agregado validado no tiene claves ni conteos faltantes. No se aplica
PCA porque se conserva la identidad categórica en una matriz dispersa. La escala
del año es una elección de preparación, no evidencia de que la tendencia prediga
bien. No se añaden rezagos ni ceros bajo supuestos no verificados de operación.

## Separación temporal

| Conjunto | Periodo | Filas verificadas |
| --- | --- | ---: |
| train | 2014-12-01 a 2025-12-31 | 1,052,704 |
| validation | 2026-01-01 a 2026-06-30 | 66,724 |
| test | 2026-07-01 a 2026-09-30 | 41,611 |

Los cortes se eligieron por calendario, no mediante comparación de errores de un
modelo. El codificador y el escalador se ajustan exclusivamente con entrenamiento.
Validación y prueba se transforman con esos mismos objetos ya ajustados.
Cambiar cortes exige ejecutar otra preparación y registrar la configuración.

La versión probada conserva las 1,161,039 filas y las 39,078,226 salidas. Se generan
406 columnas: 386 estaciones conocidas en train, 7 días de semana, 12 meses y 1 año
estandarizado. Las matrices CSR float32 ocupan aproximadamente 39.8 MiB en conjunto,
sin contar el resto de objetos del proceso. No se construye una tabla one-hot densa.

## Ejecución desde CMD

Desde la raíz de Equipo11 y con .venv activo:

```bat
python -m pip install scikit-learn scipy joblib
python -m MiBici.features --help
python -m MiBici.features --source-run-id 046501b1e5dd4aadbd7d98ea8ef05d38
```

La entrada predeterminada es el agregado de data/processed/historico/; se comprueba
su correspondencia con el reporte de calidad. La salida es data/processed/preprocesado/.
El CSV histórico de entrada no se modifica.

Se pueden definir otros cortes:

```bat
python -m MiBici.features --train-end 2025-12-31 --validation-end 2026-06-30 --source-run-id 046501b1e5dd4aadbd7d98ea8ef05d38
```

Para calcular sin registrar MLflow, añadir --no-mlflow. Los tres conjuntos deben
contener filas. Los cortes y transformaciones se documentan en el reporte.

## Archivos generados

- train.csv, validation.csv y test.csv: identificadores, calendario y objetivo; el
  orden de las filas coincide con sus respectivas matrices y etiquetas.
- X_train.npz, X_validation.npz, X_test.npz: matrices dispersas de predictores.
- y_train.npy, y_validation.npy, y_test.npy: conteos objetivo sin escalar.
- preprocessor.joblib: transformador ajustado con train para reutilizarlo.
- columnas_transformadas.json: nombres y orden exacto de las columnas de X.
- reporte_preprocesamiento.json: configuración, conteos, cobertura, categorías
  desconocidas, controles y versiones de software.
- manifest_sha256.json: huellas de los archivos de salida.
- mlflow_preprocessing_run.json: ID de la nueva ejecución cuando se registra.

Ejemplo de lectura posterior de una matriz:

```python
from scipy import sparse
import numpy as np
X = sparse.load_npz('data/processed/preprocesado/X_train.npz')
y = np.load('data/processed/preprocesado/y_train.npy')
assert X.shape[0] == len(y)
```

## MLflow y versiones

Se crea un nuevo run preprocesamiento-calendario-mibici en mibici-fase1-historico,
con la etiqueta dataset_version=calendar_onehot_v1. Sus entradas son los tres
splits con contexto train, validation y test. Sus artefactos conservan CSV, matrices,
etiquetas, transformador, manifiesto y reporte.

La etiqueta preparation_run_id apunta al run del agregado limpio. Su parámetro
output_sha256 se compara con el archivo de entrada antes de registrar la relación.
El EDA describe ese mismo agregado sin cambiarlo; esta etapa crea una representación
con variables y matrices transformadas. No es necesario registrar todo en un único run.

## Validación y límites

```bat
python -m unittest discover -s tests -p test_preprocessing.py -v
```

Cuatro pruebas comprueban calendario, conservación de filas y objetivo, cortes
sin solapamiento, ajuste solo con train, independencia respecto al objetivo,
estaciones nuevas, recuperación del transformador y alineación de matrices/etiquetas.
También se probó en un almacén separado el registro de tres entradas en MLflow,
la recuperación de una matriz y el rechazo de una huella fuente incompatible.
La prueba de todo el histórico se hizo con Python 3.12.14; al ejecutar desde la
terminal del equipo se registran sus propias versiones de Python y librerías.

Los estación-días ausentes siguen ausentes: esto prepara el conjunto observado,
no un panel de todas las estaciones operativas. Hay 55 estaciones de validación y
99 de prueba no vistas en train; handle_unknown=ignore deja en cero únicamente
su grupo one-hot de estación, conserva el calendario y evita errores. Esas filas
se contabilizan; no son viajes con cero salidas. Un modelo posterior debe evaluar
por separado estaciones nuevas y conocidas.

El EDA previo cubrió todo el periodo de forma retrospectiva. No se afirma una
estimación de desempeño fuera de muestra: al construir modelos deberán fijarse
las decisiones sin optimizar sobre prueba y usar una evaluación temporal adecuada.
No se aplicaron reducciones de dimensionalidad ni imputaciones por cumplir una lista.

## Estado local

El usuario ejecutó el módulo con Python 3.13.1 y registró el preprocesamiento en
su MLflow local con run_id 56c4baf9fd59436ea825033851e07ff3. La salida real está
resumida en reports/preprocesamiento/resultado_ejecucion_local.json; la validación
previa del entorno de prueba se conserva separada en validacion_prueba.json.
Los cambios siguen locales, pendientes de commit y push.
