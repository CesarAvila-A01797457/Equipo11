"""Compara el modelo y el servicio en un entorno limpio con resultados de referencia."""

import argparse
from datetime import date
import json
import math
from pathlib import Path
import platform
from importlib.metadata import version
from urllib.request import Request, urlopen

import numpy as np

from MiBici.api import build_prediction_features, load_registered_model
from MiBici.config import PROJ_ROOT, PROCESSED_DATA_DIR
from MiBici.features import (
    PREDICTORS,
    build_calendar_features,
    load_daily_counts,
    sha256,
    temporal_split,
)
from MiBici.modeling.train import calculate_metrics

DEFAULT_MODEL_URI = "models:/mibici-poisson/2"
DEFAULT_REFERENCE = PROJ_ROOT / "reports/fase3/reproducibility_reference.json"
DEFAULT_INPUT = (
    PROCESSED_DATA_DIR / "historico_filtrado/mibici_salidas_estacion_fecha.csv"
)
DEFAULT_QUALITY_REPORT = (
    PROCESSED_DATA_DIR / "historico_filtrado/mibici_quality_report.json"
)
DEFAULT_OUTPUT = PROJ_ROOT / "reports/fase3/reproducibility_clean_env.json"


def compare_values(actual, expected, tolerance):
    """Compara listas de números con una tolerancia absoluta."""
    return len(actual) == len(expected) and all(
        math.isclose(float(left), float(right), rel_tol=0, abs_tol=tolerance)
        for left, right in zip(actual, expected, strict=True)
    )


def request_api_prediction(api_url, payload):
    """Consulta el servicio activo sin crear corridas de MLflow."""
    request = Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def run_check(
    model_uri,
    reference_path,
    input_path,
    quality_report_path,
    output_path,
    api_url,
):
    """Evalúa el artefacto y el endpoint contra métricas y salidas de referencia."""
    reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    tolerance = reference["tolerance"]

    dataset_hash = sha256(input_path)
    data = load_daily_counts(input_path, quality_report_path)
    frame = build_calendar_features(data)
    splits = temporal_split(
        frame,
        reference["split"]["train_end"],
        reference["split"]["validation_end"],
    )

    model = load_registered_model(model_uri)
    test = splits["test"]
    test_predictions = np.asarray(model.predict(test[PREDICTORS]), dtype=float)
    metrics = calculate_metrics(test["salidas"].to_numpy(), test_predictions)

    sample = reference["api_sample"]
    sample_request = sample["request"]
    sample_features = build_prediction_features(
        date.fromisoformat(sample_request["fecha_objetivo"]),
        sample_request["estacion_ids"],
    )
    direct_predictions = np.asarray(model.predict(sample_features), dtype=float).tolist()
    api_response = request_api_prediction(api_url, sample_request)
    api_predictions = [
        item["salidas_estimadas"] for item in api_response["predicciones"]
    ]
    expected_predictions = sample["salidas_estimadas"]
    reference_packages = reference["packages"]
    current_packages = {
        name: version(name)
        for name in reference_packages
    }

    comparisons = {
        "model_uri_matches": model_uri == reference["model_uri"],
        "python_version_matches": platform.python_version() == reference["python"],
        "package_versions_match": current_packages == reference_packages,
        "dataset_hash_matches": dataset_hash == reference["dataset_sha256"],
        "test_rows_match": len(test) == reference["test_rows"],
        "test_mae_matches": math.isclose(
            metrics["mae"], reference["metrics"]["test_mae"],
            rel_tol=0, abs_tol=tolerance,
        ),
        "test_bias_matches": math.isclose(
            metrics["bias"], reference["metrics"]["test_bias"],
            rel_tol=0, abs_tol=tolerance,
        ),
        "model_predictions_match": compare_values(
            direct_predictions, expected_predictions, tolerance
        ),
        "api_predictions_match": compare_values(
            api_predictions, expected_predictions, tolerance
        ),
    }

    report = {
        "model_uri": model_uri,
        "reference_run_id": reference["mlflow_run_id"],
        "python": platform.python_version(),
        "packages": current_packages,
        "dataset_sha256": dataset_hash,
        "test_rows": len(test),
        "metrics": {"test_mae": metrics["mae"], "test_bias": metrics["bias"]},
        "api_sample": {
            "request": sample_request,
            "direct_model_predictions": direct_predictions,
            "api_predictions": api_predictions,
        },
        "tolerance": tolerance,
        "comparisons": comparisons,
        "reproducible": all(comparisons.values()),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    print(f"Modelo: {model_uri}")
    print(f"MAE en prueba: {metrics['mae']:.6f}")
    print(f"Sesgo en prueba: {metrics['bias']:+.6f}")
    print(f"Predicciones de la API: {api_predictions}")
    print(f"Comparaciones: {comparisons}")
    print(f"Reporte: {output_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-uri", default=DEFAULT_MODEL_URI)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--quality-report", type=Path, default=DEFAULT_QUALITY_REPORT)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/predict")
    args = parser.parse_args()
    report = run_check(
        model_uri=args.model_uri,
        reference_path=args.reference,
        input_path=args.input_path,
        quality_report_path=args.quality_report,
        output_path=args.output_path,
        api_url=args.api_url,
    )
    raise SystemExit(0 if report["reproducible"] else 1)


if __name__ == "__main__":
    main()
