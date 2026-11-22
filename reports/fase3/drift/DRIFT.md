# Simulación de drift de identificadores de estación

Se evalúa un fallo sintético del pipeline de entrada que cambia IDs de estación por códigos desconocidos para el modelo. Se mantienen las fechas, los conteos reales y el orden de las observaciones. El modelo queda congelado.

La referencia usa validación de enero a junio de 2026. Los escenarios alteran subconjuntos anidados con semilla fija. La fracción se aplica solo a las filas cuya estación es conocida por el preprocesador.

| Fracción alterada | Filas alteradas | TV | MAE | Sesgo | Cambio MAE | Alerta conjunta |
|---|---:|---:|---:|---:|---:|---|
| 0% | 0 | 0.000 | 12.0651 | +8.2518 | +0.0% | False |
| 10% | 6575 | 0.099 | 12.9126 | +7.5348 | +7.0% | False |
| 30% | 19726 | 0.296 | 14.7114 | +5.9690 | +21.9% | True |
| 60% | 39452 | 0.591 | 17.3618 | +3.7384 | +43.9% | True |
| 100% | 65754 | 0.986 | 20.9201 | +0.7993 | +73.4% | True |

Se propone alertar con TV >= 0.10 y con un incremento relativo del MAE >= 10%. Son umbrales educativos definidos antes de ejecutar la simulación. La alerta conjunta requiere que se cumplan ambos criterios.

Ante drift se revisaría el catálogo y la correspondencia de IDs. Ante degradación se investigaría su causa. Una vez corregidas las entradas, se volvería a evaluar. El reentrenamiento se consideraría si el cambio persistiera con datos válidos y nuevas etiquetas.

El drift de entrada se puede medir sin etiquetas. La pérdida de desempeño requiere conteos observados. Esta prueba representa corrupción de una variable de entrada y no permite concluir que exista drift real en MiBici ni que los IDs sintéticos representen estaciones nuevas.

La comparación corresponde a validación y es independiente de las métricas de prueba publicadas en la Fase 2.