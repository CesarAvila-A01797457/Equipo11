# Ejecución del histórico con filtro de viajes cortos

Fecha: 25 de octubre de 2026.

Esta ejecución incorpora la regla del PR #7 al procesamiento por bloques del PR #6. Los resultados anteriores, en `reports/eda_historico` y `reports/EJECUCION_LOCAL.md`, corresponden al histórico sin este filtro. Se conservan como evidencia de esa versión; las cifras de ambas versiones no deben mezclarse en la presentación.

## Regla de limpieza

1. Validar identificador de viaje, fecha de inicio y estación de origen.
2. Conservar la primera fila válida por `(mes de inicio, Viaje_Id)`, incluso entre archivos y bloques.
3. Excluir viajes con origen igual a destino y duración mayor o igual a cero y menor a 120 segundos. Un viaje de exactamente 120 segundos se conserva.

Una fecha de fin inválida, una duración negativa o un destino inválido no activan el filtro. La regla es una decisión de limpieza del equipo: no demuestra que todos los viajes excluidos fueran pruebas.

Los identificadores de viajes excluidos por duración siguen reservados para evitar que un duplicado posterior vuelva a contarse. El script reutiliza el filtro modular del compañero y conserva la deduplicación global en SQLite temporal. Los CSV originales no se modifican.

## Comparación de versiones

| Medida | Sin filtro de duración | Con filtro de duración |
|---|---:|---:|
| Archivos mensuales | 142 | 142 |
| Viajes leídos | 39,078,984 | 39,078,984 |
| Duplicados excluidos | 758 | 758 |
| Viajes cortos excluidos | 0 (regla no aplicada) | 1,352,390 |
| Viajes conservados | 39,078,226 | 37,725,836 |
| Filas estación-fecha | 1,161,039 | 1,160,636 |

El filtro reduce los viajes conservados aproximadamente 3.46 % respecto a la versión anterior. Desaparecen 403 filas estación-fecha porque no quedan viajes retenidos en esos grupos; el agregado no genera filas explícitas con cero salidas.

## EDA filtrado

Periodo: 2014-12-01 a 2026-09-30; 4,322 días con datos, 486 identificadores de estación observados y ninguna fecha completa ausente. En el agregado no se detectaron valores obligatorios faltantes, claves estación-fecha duplicadas ni fechas, estaciones o conteos inválidos. Los totales coinciden con el reporte de preparación.

Media diaria del sistema: 8,728.79 salidas. Mediana por estación-fecha: 23; percentil 99: 167; máximo: 546. El martes tiene la mayor media de salidas del sistema. Las diez estaciones principales concentran aproximadamente 13.63 % de las salidas. Los años 2014 y 2026 son parciales.

Las alertas IQR son diagnósticas y no eliminan picos de demanda. El EDA es descriptivo del histórico completo; las transformaciones para modelado se ajustan exclusivamente con entrenamiento.

Resultados: `reports/eda_historico_filtrado/`, con ocho figuras, tablas, reporte y metadatos de MLflow. El notebook y HTML anteriores corresponden a la versión sin filtro; estos comandos no los regeneran.

## Preprocesamiento filtrado

| Partición | Periodo | Filas | Salidas |
|---|---|---:|---:|
| Entrenamiento | 2014-12-01 a 2025-12-31 | 1,052,321 | 34,498,507 |
| Validación | 2026-01-01 a 2026-06-30 | 66,715 | 2,013,146 |
| Prueba | 2026-07-01 a 2026-09-30 | 41,600 | 1,214,183 |

Se conservan 1,160,636 filas y 37,725,836 salidas. Los predictores son estación, día de semana, mes y año. Se aplica one-hot a estación, día de semana y mes, y estandarización al año, con 406 columnas finales en matrices dispersas. El transformador se ajusta únicamente con entrenamiento; la variable objetivo `salidas` no se escala ni se incluye entre los predictores.

Validación contiene 55 estaciones no vistas en entrenamiento (961 filas), y prueba contiene 99 (7,875 filas). Su grupo de estación en one-hot queda en cero; las demás variables conservan su representación. No se imputan ausencias como cero, no se aplica PCA y no se entrena un modelo en esta fase.

Artefactos completos: `data/processed/preprocesado_filtrado/`, excluidos de Git y registrados en MLflow. Se conserva una copia pequeña del reporte y de los metadatos de la ejecución en `reports/preprocesamiento_filtrado/`.

## Trazabilidad en MLflow

Experimento: `mibici-fase1-historico-filtrado`.

| Etapa | Run ID |
|---|---|
| Preparación | `ec55a8421f944f4a913bc064975ca289` |
| EDA | `e60748c057924948a74f60feae1aa2b9` |
| Preprocesamiento | `46158363caa24cd19ee5479cd90ee52c` |

EDA y preprocesamiento referencian directamente la ejecución de preparación. Ambos usan el mismo CSV agregado, con SHA-256 `8bdfbd5c8952f9f261d4dbf950a48756cdc2b56c9c686a4898b63aac95beae83`.

El almacén de MLflow es local. Publicar este documento en GitHub no publica la base de datos ni todos los artefactos de MLflow; para la entrega se deben acompañar los resultados con capturas de la interfaz local.

## Reproducción desde la raíz del repositorio

Con el entorno activado, dependencias instaladas y CSV originales disponibles:

```bat
python -m MiBici.prepare_data_streaming --chunk-size 25000 --output-dir data/processed/historico_filtrado --experiment-name mibici-fase1-historico-filtrado
```

Las siguientes ejecuciones deben usar el nuevo identificador emitido por preparación en lugar de `RUN_ID_PREPARACION`:

```bat
python -m MiBici.eda --input-path data/processed/historico_filtrado/mibici_salidas_estacion_fecha.csv --quality-report data/processed/historico_filtrado/mibici_quality_report.json --output-dir reports/eda_historico_filtrado --experiment-name mibici-fase1-historico-filtrado --source-run-id RUN_ID_PREPARACION
python -m MiBici.features --input-path data/processed/historico_filtrado/mibici_salidas_estacion_fecha.csv --quality-report data/processed/historico_filtrado/mibici_quality_report.json --output-dir data/processed/preprocesado_filtrado --experiment-name mibici-fase1-historico-filtrado --source-run-id RUN_ID_PREPARACION
python -m unittest discover -s tests -v
```

Validación local reportada: 11 pruebas aprobadas, incluyendo equivalencia entre la preparación completa y por bloques con distintos tamaños de bloque, límites de duración, duplicados entre archivos y preservación de la salida anterior cuando todos los viajes resultan excluidos.
