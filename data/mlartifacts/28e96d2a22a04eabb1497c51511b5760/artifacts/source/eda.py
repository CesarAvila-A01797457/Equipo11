"""EDA reproducible de salidas diarias MiBici; no modifica ni imputa los datos."""

import argparse
import calendar
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DAYS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, data):
    Path(path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def load_and_validate(input_path, quality_report=None):
    required = ["fecha", "estacion_id", "salidas"]
    data = pd.read_csv(input_path, usecols=required, dtype="string")
    if data.empty:
        raise ValueError("El agregado está vacío")
    missing = data.isna().sum().astype(int).to_dict()
    if any(missing.values()):
        raise ValueError(f"Hay campos obligatorios faltantes: {missing}")
    data["fecha"] = pd.to_datetime(data.fecha, format="%Y-%m-%d", errors="coerce")
    if data.fecha.isna().any():
        raise ValueError("Hay fechas no interpretables")
    for column, minimum in [("estacion_id", 1), ("salidas", 0)]:
        values = pd.to_numeric(data[column], errors="coerce")
        if (values.isna() | values.lt(minimum) | values.mod(1).ne(0)).any():
            raise ValueError(f"Hay valores inválidos en {column}")
        data[column] = values.astype("int64")
    duplicates = int(data.duplicated(["fecha", "estacion_id"]).sum())
    if duplicates:
        raise ValueError(f"Hay {duplicates} claves estación-fecha duplicadas")
    if int(data.salidas.sum()) == 0:
        raise ValueError(
            "El dataset no contiene salidas; no permite analizar concentración de uso"
        )
    quality = None
    if quality_report is not None:
        quality = json.loads(Path(quality_report).read_text(encoding="utf-8-sig"))
        if quality["rows_kept"] != int(data.salidas.sum()) or quality[
            "output_rows"
        ] != len(data):
            raise ValueError(
                "El reporte de preparación no coincide con el CSV agregado"
            )
        if quality["rows_read"] != quality["rows_kept"] + quality["rows_removed_total"]:
            raise ValueError("Los conteos del reporte de preparación no concilian")
    diagnostics = {
        "missing_values": missing,
        "duplicate_station_date_keys": duplicates,
        "invalid_dates": 0,
        "invalid_station_ids": 0,
        "invalid_counts": 0,
        "explicit_zero_rows": int(data.salidas.eq(0).sum()),
        "matches_preparation_report": quality is not None,
    }
    return data, diagnostics, quality


def build_tables(data):
    dates = pd.date_range(data.fecha.min(), data.fecha.max(), freq="D")
    daily = data.groupby("fecha").agg(
        salidas=("salidas", "sum"), estaciones_presentes=("estacion_id", "nunique")
    )
    positive = data[data.salidas.gt(0)]
    coverage = positive.groupby("fecha").estacion_id.nunique()
    daily["estaciones_con_viajes"] = coverage.reindex(daily.index, fill_value=0)
    daily = daily.reindex(dates).rename_axis("fecha")
    daily["salidas_por_estacion_con_viajes"] = (
        daily.salidas / daily.estaciones_con_viajes.replace(0, np.nan)
    )
    # Fechas ausentes quedan como NaN: no se inventan ceros.
    observed = daily.dropna(subset=["salidas"])
    monthly = observed.groupby(observed.index.to_period("M")).agg(
        salidas=("salidas", "sum"),
        dias_con_datos=("salidas", "count"),
        promedio_diario=("salidas", "mean"),
        promedio_estaciones_con_viajes=("estaciones_con_viajes", "mean"),
        promedio_salidas_por_estacion_con_viajes=(
            "salidas_por_estacion_con_viajes",
            "mean",
        ),
    )
    monthly.index = monthly.index.astype(str)
    monthly.index.name = "mes"
    yearly = observed.groupby(observed.index.year).agg(
        salidas=("salidas", "sum"),
        dias_con_datos=("salidas", "count"),
        promedio_diario=("salidas", "mean"),
    )
    yearly.index.name = "anio"
    yearly["dias_del_anio"] = [
        366 if calendar.isleap(int(y)) else 365 for y in yearly.index
    ]
    yearly["anio_completo"] = yearly.dias_con_datos.eq(yearly.dias_del_anio)
    weekday = observed.groupby(observed.index.day_of_week).agg(
        dias_con_datos=("salidas", "count"),
        promedio_salidas_sistema=("salidas", "mean"),
        promedio_por_estacion_con_viajes=("salidas_por_estacion_con_viajes", "mean"),
    )
    weekday = weekday.reindex(range(7)).rename_axis("dia_semana")
    weekday["nombre"] = DAYS
    station = data.groupby("estacion_id").agg(
        salidas_totales=("salidas", "sum"),
        dias_presentes=("salidas", "size"),
        promedio_en_dias_presentes=("salidas", "mean"),
        mediana=("salidas", "median"),
        primera_fecha_observada=("fecha", "min"),
        ultima_fecha_observada=("fecha", "max"),
    )
    quartiles = data.groupby("estacion_id").salidas.quantile([0.25, 0.75]).unstack()
    station["q1"] = quartiles[0.25]
    station["q3"] = quartiles[0.75]
    station["limite_iqr_superior"] = station.q3 + 1.5 * (station.q3 - station.q1)
    station["limite_iqr_inferior"] = station.q1 - 1.5 * (station.q3 - station.q1)
    flags = data.salidas.gt(
        data.estacion_id.map(station.limite_iqr_superior)
    ) | data.salidas.lt(data.estacion_id.map(station.limite_iqr_inferior))
    station["alertas_iqr"] = flags.groupby(data.estacion_id).sum().astype(int)
    station["porcentaje_alertas_iqr"] = (
        station.alertas_iqr / station.dias_presentes * 100
    )
    station = station.sort_values("salidas_totales", ascending=False)
    relations = []
    for lag in [1, 7, 28]:
        pairs = pd.concat(
            [daily.salidas, daily.salidas.shift(lag).rename("pasado")], axis=1
        ).dropna()
        can_correlate = (
            len(pairs) >= 2
            and pairs.salidas.nunique() > 1
            and pairs.pasado.nunique() > 1
        )
        relations.append(
            {
                "rezago_dias": lag,
                "pares": len(pairs),
                "pearson": pairs.salidas.corr(pairs.pasado)
                if can_correlate
                else np.nan,
                "spearman": pairs.salidas.corr(pairs.pasado, method="spearman")
                if can_correlate
                else np.nan,
            }
        )
    descriptive = data.salidas.describe(
        percentiles=[0.25, 0.5, 0.75, 0.95, 0.99]
    ).to_frame("salidas")
    return {
        "diario": daily,
        "mensual": monthly,
        "anual": yearly,
        "dia_semana": weekday,
        "estaciones": station,
        "relaciones_temporales": pd.DataFrame(relations),
        "estadisticas": descriptive,
    }


def build_summary(data, tables, diagnostics, quality):
    daily = tables["diario"]
    station = tables["estaciones"]
    weekday = tables["dia_semana"]
    q1, q3 = data.salidas.quantile([0.25, 0.75])
    low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    summary = {
        "period_start": data.fecha.min().strftime("%Y-%m-%d"),
        "period_end": data.fecha.max().strftime("%Y-%m-%d"),
        "station_date_rows": len(data),
        "total_departures": int(data.salidas.sum()),
        "stations_observed": int(data.estacion_id.nunique()),
        "calendar_days": len(daily),
        "days_with_data": int(daily.salidas.notna().sum()),
        "missing_whole_dates": int(daily.salidas.isna().sum()),
        "mean_daily_departures": float(daily.salidas.mean()),
        "median_station_date_departures": float(data.salidas.median()),
        "p99_station_date_departures": float(data.salidas.quantile(0.99)),
        "max_station_date_departures": int(data.salidas.max()),
        "min_stations_with_trips_per_day": int(daily.estaciones_con_viajes.min()),
        "max_stations_with_trips_per_day": int(daily.estaciones_con_viajes.max()),
        "weekday_highest_mean_system_departures": DAYS[
            int(weekday.promedio_salidas_sistema.idxmax())
        ],
        "top_10_station_departure_share": float(
            station.salidas_totales.head(10).sum() / data.salidas.sum()
        ),
        "pooled_iqr_lower": float(low),
        "pooled_iqr_upper": float(high),
        "pooled_iqr_alert_rows": int(
            (data.salidas.lt(low) | data.salidas.gt(high)).sum()
        ),
        "within_station_iqr_alert_rows": int(station.alertas_iqr.sum()),
        "incomplete_years": [
            int(y) for y in tables["anual"].index[~tables["anual"].anio_completo]
        ],
        "diagnostics": diagnostics,
        "scope": "EDA descriptiva del histórico completo; incluye periodo posterior al futuro corte de entrenamiento.",
    }
    if quality is not None:
        summary["preparation"] = {
            key: quality[key]
            for key in [
                "input_files",
                "rows_read",
                "rows_kept",
                "rows_removed_total",
                "rows_removed",
            ]
        }
    return summary


def generate_figures(data, tables, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.figsize": (11, 5),
        }
    )
    paths = []

    def finish(fig, name):
        fig.tight_layout()
        path = output_dir / f"{name}.png"
        fig.savefig(path, dpi=150, facecolor="white")
        plt.close(fig)
        paths.append(path)

    daily, yearly = tables["diario"], tables["anual"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(daily.index, daily.salidas, color="#195b91", linewidth=0.7)
    axes[0].set(title="Salidas diarias del sistema MiBici", ylabel="Viajes por día")
    axes[1].plot(
        daily.index, daily.estaciones_con_viajes, color="#147d64", linewidth=0.8
    )
    axes[1].set(ylabel="Estaciones con viajes", xlabel="Fecha")
    finish(fig, "01_serie_y_cobertura")

    fig, ax = plt.subplots()
    bars = ax.bar(range(len(yearly)), yearly.salidas / 1_000_000, color="#195b91")
    for bar, complete in zip(bars, yearly.anio_completo):
        if not complete:
            bar.set_hatch("///")
    ax.set_xticks(
        range(len(yearly)),
        [str(y) + ("*" if not c else "") for y, c in yearly.anio_completo.items()],
    )
    ax.set(
        title="Salidas por año (* periodo incompleto)",
        ylabel="Millones de viajes",
        xlabel="Año",
    )
    finish(fig, "02_totales_anuales")

    weekday = tables["dia_semana"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, column, title in zip(
        axes,
        ["promedio_salidas_sistema", "promedio_por_estacion_con_viajes"],
        ["Promedio diario del sistema", "Promedio diario por estación con viajes"],
    ):
        ax.bar(range(7), weekday[column], color="#195b91")
        ax.set_xticks(range(7), [d[:3] for d in DAYS])
        ax.set(title=title, ylabel="Salidas")
    finish(fig, "03_patron_semanal")

    monthly = tables["mensual"].reset_index()
    periods = pd.PeriodIndex(monthly.mes, freq="M")
    monthly["anio"], monthly["numero_mes"] = periods.year, periods.month
    heat = monthly.pivot(
        index="anio", columns="numero_mes", values="promedio_diario"
    ).reindex(columns=range(1, 13))
    fig, ax = plt.subplots(figsize=(11, 7))
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#dedede")
    im = ax.imshow(np.ma.masked_invalid(heat.to_numpy()), aspect="auto", cmap=cmap)
    ax.set_xticks(
        range(12),
        [
            "Ene",
            "Feb",
            "Mar",
            "Abr",
            "May",
            "Jun",
            "Jul",
            "Ago",
            "Sep",
            "Oct",
            "Nov",
            "Dic",
        ],
    )
    ax.set_yticks(range(len(heat)), heat.index)
    ax.set(
        title="Promedio diario por mes y año (gris: sin datos)",
        xlabel="Mes",
        ylabel="Año",
    )
    fig.colorbar(im, ax=ax, label="Viajes diarios del sistema")
    finish(fig, "04_calendario_mensual")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(data.salidas, bins=70, color="#195b91")
    axes[0].set_yscale("log")
    axes[0].set(
        title="Distribución de salidas presentes",
        xlabel="Salidas por estación-día",
        ylabel="Frecuencia (escala log)",
    )
    probabilities = np.linspace(0, 1, 501)
    axes[1].plot(data.salidas.quantile(probabilities), probabilities, color="#195b91")
    axes[1].set(
        title="Distribución acumulada resumida por cuantiles",
        xlabel="Salidas por estación-día",
        ylabel="Proporción acumulada",
    )
    finish(fig, "05_distribucion_y_ecdf")

    top = tables["estaciones"].head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(top.index.astype(str), top.salidas_totales, color="#195b91")
    ax.set(
        title="Quince estaciones con más salidas acumuladas",
        xlabel="Viajes en todo el periodo",
        ylabel="ID de estación (categoría)",
    )
    finish(fig, "06_estaciones_principales")

    fig, ax = plt.subplots()
    monthly_dates = pd.to_datetime(tables["mensual"].index + "-01")
    ax.plot(
        monthly_dates,
        tables["mensual"].promedio_salidas_por_estacion_con_viajes,
        color="#147d64",
    )
    ax.set(
        title="Promedio diario por estación con viajes, resumido por mes",
        xlabel="Fecha",
        ylabel="Salidas / estaciones con viajes",
    )
    finish(fig, "07_uso_normalizado_por_cobertura")

    pairs = pd.concat(
        [daily.salidas.rename("actual"), daily.salidas.shift(7).rename("anterior")],
        axis=1,
    ).dropna()
    fig, ax = plt.subplots()
    ax.scatter(
        pairs.anterior, pairs.actual, s=7, alpha=0.3, color="#195b91", rasterized=True
    )
    ax.set(
        title="Relación descriptiva con las salidas de siete días antes",
        xlabel="Salidas del sistema en t − 7 días",
        ylabel="Salidas del sistema en t",
    )
    finish(fig, "08_relacion_temporal")
    return paths


def write_report(summary, tables, output_dir):
    s = summary
    yearly = tables["anual"]
    annual_rows = "\n".join(
        f"| {int(y)} | {int(r.salidas):,} | {int(r.dias_con_datos)} | {'Sí' if r.anio_completo else 'No'} |"
        for y, r in yearly.iterrows()
    )
    preparation = s.get("preparation")
    reconciliation = (
        (
            f"El reporte de preparación registra {preparation['rows_read']:,} viajes originales, "
            f"{preparation['rows_removed_total']:,} exclusiones y {preparation['rows_kept']:,} viajes retenidos. "
            "La suma de salidas y la cantidad de filas coinciden con ese reporte."
        )
        if preparation
        else "No se proporcionó reporte de preparación para conciliar los conteos."
    )
    temporal_rows = "\n".join(
        f"| {int(r.rezago_dias)} | {int(r.pares):,} | {r.pearson:.3f} | {r.spearman:.3f} |"
        for _, r in tables["relaciones_temporales"].iterrows()
    )
    text = f"""# Análisis exploratorio del histórico MiBici

Se analiza el uso realizado de MiBici por estación y día para preparar un futuro
modelo de salidas. El agregado cubre {s["period_start"]} a {s["period_end"]}:
{s["total_departures"]:,} salidas, {s["station_date_rows"]:,} estación-días presentes
y {s["stations_observed"]} IDs de estación distintos. La variación temporal y de
cobertura exige considerar calendario e historia anterior, sin confundir viajes
observados con disponibilidad de bicicletas o demanda insatisfecha.

## 1 Diagnóstico y conciliación

{reconciliation}

Se validaron columnas, fechas, IDs positivos, conteos enteros no negativos y
unicidad de la clave estación-fecha. Hay {s["diagnostics"]["explicit_zero_rows"]:,}
filas con cero explícito. En un calendario de {s["calendar_days"]:,} días,
{s["days_with_data"]:,} tienen datos y {s["missing_whole_dates"]:,} no tienen observaciones
del sistema. Las fechas ausentes se conservan como faltantes; no se imputan a cero.

## 2 Evolución y cobertura

El promedio del sistema es {s["mean_daily_departures"]:,.1f} viajes por día con datos.
El número diario de estaciones con viajes oscila entre
{s["min_stations_with_trips_per_day"]} y {s["max_stations_with_trips_per_day"]}.
Los cambios del total deben interpretarse junto con esa cobertura. Una estación
sin viajes no puede clasificarse como cerrada con estos datos.

![Serie y cobertura](figures/01_serie_y_cobertura.png)

| Año | Salidas | Días con datos | Año calendario completo |
| --- | ---: | ---: | --- |
{annual_rows}

![Totales anuales](figures/02_totales_anuales.png)

Los años incompletos se señalan con asterisco y trama. Sus totales no son directamente
comparables con los de un año completo. Los cambios entre años también pueden
reflejar crecimiento de la red y diferencias de captura; no se establece causalidad.

## 3 Patrones de calendario

El mayor promedio de viajes del sistema corresponde a
**{s["weekday_highest_mean_system_departures"]}**. Es una asociación del periodo
completo, que mezcla años y configuraciones de red. Se presenta también el cociente
diario de salidas entre estaciones con viajes para observar el efecto de cobertura.

![Patrón semanal](figures/03_patron_semanal.png)
![Calendario mensual](figures/04_calendario_mensual.png)

El mapa de calor usa promedio diario, en lugar de comparar meses de distinta
duración por su total. El gris representa ausencia de datos, no cero viajes.

## 4 Distribución y posibles atípicos

La mediana por estación-día presente es {s["median_station_date_departures"]:.1f};
el percentil 99 es {s["p99_station_date_departures"]:.1f} y el máximo es
{s["max_station_date_departures"]:,}. La distribución debe revisarse por estación y
periodo antes de elegir una función de pérdida o una transformación del objetivo.

![Distribución y ECDF](figures/05_distribucion_y_ecdf.png)

La regla IQR agrupada señala {s["pooled_iqr_alert_rows"]:,} filas fuera de
[{s["pooled_iqr_lower"]:.1f}, {s["pooled_iqr_upper"]:.1f}]. Al calcularla dentro de cada
estación aparecen {s["within_station_iqr_alert_rows"]:,} alertas. Son diagnósticos
descriptivos: **no se elimina ninguna fila**. La regla agrupada mezcla estaciones
con niveles de uso y periodos diferentes; un conteo alto puede ser uso real.

## 5 Diferencias entre estaciones

Las diez estaciones con mayor volumen acumulan {s["top_10_station_departure_share"]:.1%}
de las salidas. El ranking histórico mezcla intensidad de uso y tiempo observado;
la tabla estaciones.csv incluye días presentes y promedio condicionado a esos días.
Primera y última fecha observadas no prueban fechas de apertura o cierre.

![Estaciones principales](figures/06_estaciones_principales.png)
![Uso por cobertura](figures/07_uso_normalizado_por_cobertura.png)

El denominador del cociente son estaciones **con viajes**, no estaciones activas.
Por tanto, este cociente no estima demanda por estación operativa ni resuelve
por sí mismo cambios de composición de la red.

## 6 Relaciones temporales

| Rezago en días | Pares disponibles | Pearson | Spearman |
| --- | ---: | ---: | ---: |
{temporal_rows}

![Relación temporal](figures/08_relacion_temporal.png)

Los rezagos corresponden a fechas exactas del calendario. Las correlaciones del
total del sistema pueden reflejar tendencia, cobertura y calendario compartidos;
no demuestran capacidad predictiva fuera de muestra y no se interpretan p-valores.
No se correlaciona el ID de estación con las salidas, porque es una categoría.

## 7 Consecuencias para el preprocesamiento

1. Separar por fecha antes de ajustar transformaciones aprendidas de los datos.
2. Representar estación como categoría y calendario mediante variables justificadas.
3. Construir rezagos y medias exclusivamente con información disponible antes del objetivo.
4. Evaluar estaciones nuevas y cambios de cobertura por separado.
5. Resolver la representación de ceros con información histórica de operación o una
   suposición explícita; no completar automáticamente toda estación en todo día.
6. Justificar normalización según el modelo. No aplicar PCA ni borrar picos de uso
   automáticamente solo por ser técnicas disponibles.

## 8 Alcance y trazabilidad

Este EDA no cambia, limpia de nuevo, normaliza ni completa el dataset. Se conserva
la huella SHA-256 del CSV, las tablas, las figuras, las versiones de librerías y el
código. El notebook utiliza funciones del módulo para evitar duplicar la lógica.

Se adaptó la estructura metodológica del notebook previo A03_A01797108 de Room
Occupancy: diagnóstico, comprobaciones, estadística univariada, ECDF, IQR y
conclusiones. No se trasladaron los límites de sensores, las clases de ocupación,
la imputación ni las comparaciones contra original.csv, que no corresponden a MiBici.

El análisis retrospectivo describe todo el histórico. Si se usa para seleccionar
variables o tomar decisiones de modelado, habrá que repetir esa selección en el
periodo de entrenamiento y reservar prueba sin usarla para ajustar el modelo.
"""
    (Path(output_dir) / "EDA_MiBici.md").write_text(text, encoding="utf-8")


def log_results(
    data,
    input_path,
    output_dir,
    summary,
    experiment_name,
    source_run_id=None,
    tracking_uri=None,
):
    import mlflow
    import mlflow.data

    tracking_uri = (
        tracking_uri or f"sqlite:///{(ROOT / 'data/mlflow.db').resolve().as_posix()}"
    )
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    if source_run_id:
        source_run = client.get_run(source_run_id)
        expected = source_run.data.params.get("output_sha256")
        if expected and expected != summary["input_sha256"]:
            raise ValueError(
                "La huella del CSV no coincide con el run de preparación indicado"
            )
    if mlflow.get_experiment_by_name(experiment_name) is None:
        location = (ROOT / "data/mlartifacts").resolve().as_uri()
        mlflow.create_experiment(experiment_name, artifact_location=location)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="eda-historico-mibici") as run:
        mlflow.set_tags(
            {"stage": "eda", "preparation_run_id": source_run_id or "not_provided"}
        )
        mlflow.log_params(
            {
                "input_sha256": summary["input_sha256"],
                "period_start": summary["period_start"],
                "period_end": summary["period_end"],
                "data_modified": False,
                "code_sha256": sha256(Path(__file__)),
            }
        )
        mlflow.log_metrics(
            {
                key: summary[key]
                for key in [
                    "station_date_rows",
                    "total_departures",
                    "stations_observed",
                    "days_with_data",
                    "missing_whole_dates",
                    "pooled_iqr_alert_rows",
                    "within_station_iqr_alert_rows",
                ]
            }
        )
        mlflow.log_input(
            mlflow.data.from_pandas(
                data,
                source=str(Path(input_path).resolve()),
                name="mibici_historico_preparado",
                targets="salidas",
            ),
            context="eda",
        )
        mlflow.log_artifacts(
            str(Path(output_dir) / "figures"), artifact_path="eda/figures"
        )
        mlflow.log_artifacts(
            str(Path(output_dir) / "tables"), artifact_path="eda/tables"
        )
        for name in ["resumen_eda.json", "EDA_MiBici.md"]:
            mlflow.log_artifact(str(Path(output_dir) / name), artifact_path="eda")
        mlflow.log_artifact(str(Path(__file__).resolve()), artifact_path="source")
        return run.info.run_id


def run_eda(
    input_path,
    output_dir,
    quality_report=None,
    log_mlflow=False,
    experiment_name="mibici-fase1-historico",
    source_run_id=None,
    tracking_uri=None,
):
    input_path, output_dir = Path(input_path).resolve(), Path(output_dir).resolve()
    if output_dir == input_path.parent or input_path.is_relative_to(output_dir):
        raise ValueError("Use una carpeta de reportes separada de los datos de entrada")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)
    print("Validando el dataset agregado...", flush=True)
    data, diagnostics, quality = load_and_validate(input_path, quality_report)
    print(f"Calculando EDA de {len(data):,} filas agregadas...", flush=True)
    tables = build_tables(data)
    summary = build_summary(data, tables, diagnostics, quality)
    summary["input_sha256"] = sha256(input_path)
    summary["source_preparation_run_id"] = source_run_id
    summary["software_versions"] = {
        name: version(name) for name in ["pandas", "numpy", "matplotlib", "scipy"]
    }
    summary["software_versions"]["python"] = platform.python_version()
    for name, table in tables.items():
        table.to_csv(
            output_dir / "tables" / f"{name}.csv", index=name != "relaciones_temporales"
        )
    missing_dates = tables["diario"].index[tables["diario"].salidas.isna()]
    pd.DataFrame({"fecha": missing_dates}).to_csv(
        output_dir / "tables/fechas_sin_observaciones.csv", index=False
    )
    save_json(output_dir / "resumen_eda.json", summary)
    generate_figures(data, tables, output_dir / "figures")
    write_report(summary, tables, output_dir)
    if log_mlflow:
        run_id = log_results(
            data,
            input_path,
            output_dir,
            summary,
            experiment_name,
            source_run_id,
            tracking_uri,
        )
        save_json(
            output_dir / "mlflow_eda_run.json",
            {
                "run_id": run_id,
                "experiment_name": experiment_name,
                "source_preparation_run_id": source_run_id,
            },
        )
        summary["eda_run_id"] = run_id
    print(f"EDA terminado. Reporte y ocho figuras en {output_dir}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-path",
        type=Path,
        default=ROOT / "data/processed/historico/mibici_salidas_estacion_fecha.csv",
    )
    parser.add_argument(
        "--quality-report",
        type=Path,
        default=ROOT / "data/processed/historico/mibici_quality_report.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "reports/eda_historico"
    )
    parser.add_argument("--experiment-name", default="mibici-fase1-historico")
    parser.add_argument("--source-run-id")
    parser.add_argument("--tracking-uri")
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Generar EDA sin registrar una nueva ejecución.",
    )
    args = parser.parse_args()
    options = vars(args)
    options["log_mlflow"] = not options.pop("no_mlflow")
    print(json.dumps(run_eda(**options), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
