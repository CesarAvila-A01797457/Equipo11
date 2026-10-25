# EDA del histórico MiBici

Se adapta la estructura de A03_A01797108.ipynb (Room Occupancy) al problema de
salidas diarias de MiBici. Se reutiliza el enfoque metodológico: diagnóstico,
validaciones, estadística univariada, ECDF, IQR, relaciones y conclusiones.
No se reutilizan los límites de sensores ni sus imputaciones.

## Ejecución desde CMD

Con el entorno .venv activo y estando en la raíz de Equipo11:

```bat
python -m pip install matplotlib
python -m MiBici.eda --source-run-id 046501b1e5dd4aadbd7d98ea8ef05d38
```

También puede usarse `python -m MiBici.plots` con las mismas opciones.
El origen predeterminado es data/processed/historico/. Los resultados quedan en
reports/eda_historico/. Para generar reportes sin crear un run adicional, usar
`python -m MiBici.eda --no-mlflow`.

El run_id anterior identifica la preparación histórica del equipo. Si se genera
otra preparación, usar su ID y comprobar que corresponde al mismo CSV. El módulo
rechaza la asociación si la huella output_sha256 del run no coincide con el archivo.

## Notebook

notebooks/01_eda_mibici_historico.ipynb contiene 20 celdas, nueve de código, y ocho
figuras embebidas. Se entregó ejecutado sin errores sobre el agregado histórico.
Para abrirlo y ejecutarlo en Jupyter, instalar opcionalmente:

```bat
python -m pip install jupyterlab ipykernel
python -m jupyterlab notebooks/01_eda_mibici_historico.ipynb
```

El notebook llama las funciones del módulo y, por defecto, no registra una nueva
corrida en MLflow. La exportación reports/eda_historico/EDA_MiBici_Historico.html
permite leer los resultados sin Jupyter y sin ejecutar código.

## Qué se genera

- Ocho figuras PNG: serie y cobertura, totales anuales, semana, mapa mensual,
  distribución y ECDF, ranking de estaciones, cociente por cobertura y rezago de siete días.
- Ocho tablas CSV, resumen JSON y reporte Markdown con resultados e interpretaciones.
- Con el registro habilitado: run eda-historico-mibici con datos asociados, huellas,
  métricas, código y artefactos de EDA. No es una versión transformada del dataset.

## Validación realizada

Se conciliaron 39,078,226 salidas y 1,161,039 filas con el reporte de preparación.
Se comprobaron 4,322 días con datos y 486 IDs históricos. No hay claves duplicadas
ni fechas completas ausentes. No hay ceros explícitos en el agregado.

Cuatro pruebas automatizadas comprueban validaciones, rechazo de reportes
incompatibles, fechas ausentes sin imputación y conservación de datos. Ejecutar:

```bat
python -m unittest discover -s tests -p test_eda.py -v
```

Además se verificaron en un almacén MLflow de prueba el registro terminado, la
recuperación de tablas, las ocho figuras y el rechazo de un run fuente con huella
incompatible. El notebook fue ejecutado en un entorno de prueba con Python 3.12.14;
las versiones usadas se registran en resumen_eda.json. Al ejecutar el comando local,
se recalculan resultados y versiones con el entorno del equipo (Python 3.13).
El usuario ejecutó el EDA con Python 3.13.1 y lo registró en su MLflow local.
Run confirmado: e3ee05e7f4f84dba8d10221cc38dd655. Ver reports/EJECUCION_LOCAL.md.

## Interpretación y límites

- Los totales describen uso realizado, no disponibilidad ni demanda insatisfecha.
- El catálogo actual y los IDs históricos tienen alcances temporales diferentes.
- Estación-días ausentes no se completan como cero. La operación histórica requiere
  catálogo o una suposición explícita, que pertenece al preprocesamiento.
- Los años 2014 y 2026 están incompletos y se señalan en la comparación anual.
- El ID de estación se trata como categoría; no se interpreta su valor como magnitud.
- Las alertas IQR no eliminan filas. No se imputan ni normalizan los datos.
- El EDA de todo el periodo es retrospectivo. La selección de variables y ajuste
  de transformaciones deben limitarse a entrenamiento al construir modelos.
- Las correlaciones de rezagos pueden reflejar tendencia y calendario, no desempeño
  predictivo fuera de muestra. No se interpretan p-valores como evidencia causal.

## Cambios locales

Se añade MiBici/eda.py, el notebook, reportes, metodología y pruebas. plots.py
queda como entrada del módulo de EDA. Se añade matplotlib explícitamente a
requirements.txt y una sección al README. El EDA conserva los CSV de entrada y el notebook de referencia. El registro
desde la terminal del usuario añade una nueva ejecución a su base local de MLflow.
Los cambios permanecen locales y pendientes de commit y push.


## Compatibilidad

Se usa day_of_week en lugar del alias obsoleto dayofweek. Si una serie no tiene
variación o no hay suficientes pares, Pearson y Spearman se conservan como
indefinidos (NaN), sin convertirlos en cero ni lanzar avisos por entradas constantes.
Esta corrección no cambia las correlaciones ya calculadas del histórico real.
