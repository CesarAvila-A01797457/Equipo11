# Ejecución local de la fase 1

El 24 de octubre de 2026 se ejecutaron en Windows, con Python 3.13.1, la descarga,
preparación del histórico, EDA y preprocesamiento. Esta evidencia corresponde a
salidas de terminal compartidas por el integrante que ejecutó el proyecto y a los
reportes generados; los identificadores apuntan a su almacén local, no a un servidor público.

## Preparación histórica

- 142 CSV mensuales de diciembre de 2014 a septiembre de 2026 y un catálogo actual.
- 39,078,984 viajes leídos; 758 duplicados excluidos; 39,078,226 viajes retenidos.
- 1,161,039 filas de salidas por estación y fecha, sin imputación de ceros.
- Run: 046501b1e5dd4aadbd7d98ea8ef05d38.
- Se usó lectura por bloques de 25,000 filas y deduplicación temporal en disco.

## EDA

- Run: e3ee05e7f4f84dba8d10221cc38dd655.
- 486 IDs históricos, 4,322 días con datos, sin fechas completas ausentes ni claves duplicadas.
- Ocho figuras, ocho tablas y reporte interpretativo.
- Mediana de 23 salidas por estación-día presente; percentil 99 de 173.
- Se conservaron alertas IQR, sin eliminar filas durante el análisis.

## Preprocesamiento

- Run: 56c4baf9fd59436ea825033851e07ff3.
- Versión: calendar_onehot_v1.
- Codificador y escalador ajustados únicamente con train.
- Entrenamiento: 1,052,704 filas; validación: 66,724; prueba: 41,611.
- 406 columnas, sin pérdida de filas ni salidas.
- Matrices dispersas, objetivos sin escalar y transformador guardados como artefactos.
- Estaciones desconocidas respecto de train: 55 en validación y 99 en prueba.

Huella del agregado de entrada compartida por las tres etapas:

```text
89b2abb060a49f70e861d95cb1fe7fe1a3aa1c1148f7052957149d731c795b51
```

## Comprobaciones y reproducción

El usuario confirmó ocho pruebas aprobadas antes del ajuste de compatibilidad.
Se añadió una novena prueba que comprueba correlaciones indefinidas para series
constantes. Ejecutar la suite actual con:

```bat
python -m unittest discover -s tests -v
```

Las pruebas anteriores a la ejecución local, realizadas con Python 3.12, se
conservan identificadas como evidencia del entorno de prueba. El notebook y HTML
entregados inicialmente fueron ejecutados allí; los reportes Markdown, CSV y JSON
se regeneraron después en el entorno local con Python 3.13.1.

Los runs y sus artefactos se guardan en data/mlflow.db y data/mlartifacts, fuera de
Git. Otro integrante debe ejecutar el pipeline para crear sus propios runs. Los
IDs cambian en cada ejecución. Para reproducir archivos idénticos se deben verificar
las huellas y utilizar las versiones fuente correspondientes.

El preprocesamiento conserva estación-días presentes; no se afirma cobertura de
todos los días operativos ni se entrenó un modelo. El PDF ejecutivo, los roles y
las contribuciones reales de todos los integrantes se documentan por separado.
