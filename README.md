# MiBici_Equipo11

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

MLOps Equipo 11 Proyecto

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         MiBici and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── MiBici   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes MiBici a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------
## Instalación y descarga de datos

El proyecto utiliza los CSV mensuales de viajes de [MiBici Open Data](https://mibici.net/en/open-data/) y el catálogo de estaciones `nomenclatura_2026_09.csv`. Los archivos originales se guardan en `data/raw/mibici/`; esta carpeta está excluida de Git para no publicar ni duplicar los datos.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m MiBici.dataset
```

El comando descarga las URLs de `references/mibici_urls_2014_2026.txt` y el catálogo de estaciones. Omite los archivos que ya existan y actualiza `data/raw/mibici/manifest.csv` con tamaño, fecha, estado y SHA-256 de cada archivo.

## Preparación de datos

Después de descargar los archivos, carga los viajes mensuales en un DataFrame y la nomenclatura en otro, sin eliminar filas o columnas, limpiar, normalizar, codificar variables, agregar o unir las tablas:

```bash
python -m MiBici.prepare_data
```
MLflow registra cada ejecución y la información de las fuentes para dar seguimiento a las versiones de los datos.

