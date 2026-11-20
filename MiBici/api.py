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

DEFAULT_MODEL_URI = "models:/mibici-poisson/1"


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


def load_registered_model(model_uri: str, tracking_uri: str | None = None):
    """Carga el pipeline registrado en MLflow una vez al iniciar el servicio."""
    import mlflow
    import mlflow.sklearn

    if tracking_uri is None:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri is None:
        database = (PROJ_ROOT / "data" / "mlflow.db").resolve()
        tracking_uri = f"sqlite:///{database.as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)
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
        model_input = pd.DataFrame(
            {
                "estacion_id": [str(station_id) for station_id in payload.estacion_ids],
                "dia_semana": [str(target_date.weekday())] * len(payload.estacion_ids),
                "mes": [str(target_date.month)] * len(payload.estacion_ids),
                "anio": [target_date.year] * len(payload.estacion_ids),
            }
        )
        try:
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
