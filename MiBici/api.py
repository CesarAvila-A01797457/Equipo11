"""API para estimar salidas diarias por estación con el modelo de MLflow."""

from contextlib import asynccontextmanager
from datetime import date
import os
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from MiBici.config import PROJ_ROOT
from MiBici.feature_engineering.calendar import CalendarStep

DEFAULT_MODEL_URI = "models/mibici-poisson-v4"
# Columnas del pipeline v1 (models:/mibici-poisson/2) para modelos sin nombres.
LEGACY_INPUTS = ["estacion_id", "dia_semana", "mes", "anio"]


class PredictionRequest(BaseModel):
    """Fecha objetivo y estaciones para las que se solicitan estimaciones."""

    fecha_objetivo: date = Field(description="Fecha en formato AAAA-MM-DD")
    estacion_ids: list[int] = Field(
        min_length=1, description="IDs de estaciones MiBici"
    )

    @field_validator("estacion_ids")
    @classmethod
    def validate_station_ids(cls, station_ids):
        if any(station_id < 1 for station_id in station_ids):
            raise ValueError("Los IDs de estación deben ser enteros positivos")
        if len(station_ids) != len(set(station_ids)):
            raise ValueError("No repitas IDs de estación en la solicitud")
        return station_ids


class StationPrediction(BaseModel):
    estacion_id: int
    salidas_estimadas: float


class PredictionResponse(BaseModel):
    fecha_objetivo: date
    predicciones: list[StationPrediction]


def build_prediction_features(target_date: date, station_ids: list[int]) -> pd.DataFrame:
    """Construye ``fecha`` y las variables de calendario como en entrenamiento.

    Incluye las entradas de ambos tipos de pipeline: ``fecha`` y ``estacion_id``
    para los conjuntos de ``MiBici.feature_engineering`` (v4), y
    ``estacion_id``, ``dia_semana``, ``mes`` y ``anio`` para el pipeline v1.
    """
    frame = pd.DataFrame(
        {
            "fecha": pd.Timestamp(target_date),
            "estacion_id": list(station_ids),
        },
        index=range(len(station_ids)),
    )
    return CalendarStep().transform(frame)


def select_model_input(model, features: pd.DataFrame) -> pd.DataFrame:
    """Entrega al modelo solo las columnas con las que se entrenó."""
    columns = getattr(model, "feature_names_in_", None)
    columns = LEGACY_INPUTS if columns is None else list(columns)
    missing = set(columns) - set(features.columns)
    if missing:
        raise ValueError(f"El modelo requiere columnas no disponibles: {sorted(missing)}")
    return features[columns]


def load_registered_model(model_uri: str, tracking_uri: str | None = None):
    """Carga el pipeline registrado en MLflow una vez al iniciar el servicio."""
    import mlflow
    import mlflow.sklearn
    from mlflow import MlflowClient

    if tracking_uri is None:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri is None:
        database = (PROJ_ROOT / "data" / "mlflow.db").resolve()
        tracking_uri = f"sqlite:///{database.as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)

    # Carga directamente el artefacto de la versión para evitar escribir en MLflow.
    if model_uri.startswith("models:/"):
        model_name, separator, version = model_uri.removeprefix("models:/").rpartition(
            "/"
        )
        if separator and version.isdigit():
            client = MlflowClient()
            registered_model = client.get_model_version(model_name, version)
            source = registered_model.source

            # MLflow 3 puede guardar la fuente como un ID de modelo registrado.
            if source.startswith("models:/m-"):
                logged_model_id = source.removeprefix("models:/")
                source = client.get_logged_model(logged_model_id).artifact_location

            return mlflow.sklearn.load_model(source)

    return mlflow.sklearn.load_model(model_uri)


def create_app(
    model: Any | None = None,
    model_uri: str | None = None,
    tracking_uri: str | None = None,
) -> FastAPI:
    """Crea la API; permite inyectar un modelo en pruebas locales."""
    selected_model_uri = model_uri or os.getenv("MIBICI_MODEL_URI", DEFAULT_MODEL_URI)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.model = (
            model
            if model is not None
            else load_registered_model(selected_model_uri, tracking_uri)
        )
        application.state.model_uri = selected_model_uri
        yield

    application = FastAPI(
        title="MiBici: predicción de salidas",
        description=(
            "Estima el número de viajes iniciados por estación para una fecha indicada."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    @application.get("/health", tags=["Estado"])
    def health(request: Request):
        if getattr(request.app.state, "model", None) is None:
            raise HTTPException(status_code=503, detail="El modelo no está disponible")
        return {"status": "ok", "model_uri": request.app.state.model_uri}

    @application.post(
        "/predict",
        response_model=PredictionResponse,
        tags=["Predicciones"],
        summary="Estimar salidas por estación y fecha",
    )
    def predict(payload: PredictionRequest, request: Request):
        model = getattr(request.app.state, "model", None)
        if model is None:
            raise HTTPException(status_code=503, detail="El modelo no está disponible")

        target_date = payload.fecha_objetivo
        features = build_prediction_features(target_date, payload.estacion_ids)
        try:
            model_input = select_model_input(model, features)
            estimates = np.asarray(model.predict(model_input), dtype=float)
        except Exception as error:
            raise HTTPException(
                status_code=500, detail="No fue posible generar las estimaciones"
            ) from error

        if (
            estimates.shape != (len(payload.estacion_ids),)
            or not np.isfinite(estimates).all()
            or (estimates < 0).any()
        ):
            raise HTTPException(
                status_code=500, detail="El modelo devolvió estimaciones inválidas"
            )

        return PredictionResponse(
            fecha_objetivo=target_date,
            predicciones=[
                StationPrediction(
                    estacion_id=station_id, salidas_estimadas=float(value)
                )
                for station_id, value in zip(payload.estacion_ids, estimates, strict=True)
            ],
        )

    return application


app = create_app()
