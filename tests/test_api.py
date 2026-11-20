from fastapi.testclient import TestClient

from MiBici.api import create_app


class ExampleModel:
    def __init__(self):
        self.input_frame = None

    def predict(self, frame):
        self.input_frame = frame.copy()
        return [12.5] * len(frame)


def test_predict_builds_model_features_and_returns_estimates():
    model = ExampleModel()
    app = create_app(model=model, model_uri="models:/example/1")

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"fecha_objetivo": "2026-11-20", "estacion_ids": [1, 2]},
        )

    assert response.status_code == 200
    assert response.json() == {
        "fecha_objetivo": "2026-11-20",
        "predicciones": [
            {"estacion_id": 1, "salidas_estimadas": 12.5},
            {"estacion_id": 2, "salidas_estimadas": 12.5},
        ],
    }
    assert model.input_frame.to_dict(orient="records") == [
        {"estacion_id": "1", "dia_semana": "4", "mes": "11", "anio": 2026},
        {"estacion_id": "2", "dia_semana": "4", "mes": "11", "anio": 2026},
    ]


def test_predict_rejects_invalid_station_ids():
    app = create_app(model=ExampleModel())

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"fecha_objetivo": "2026-11-20", "estacion_ids": [1, 1]},
        )

    assert response.status_code == 422


def test_openapi_documents_prediction_schema():
    app = create_app(model=ExampleModel())

    with TestClient(app) as client:
        schema = client.get("/openapi.json").json()

    assert "/predict" in schema["paths"]
    assert "PredictionRequest" in schema["components"]["schemas"]
    assert "PredictionResponse" in schema["components"]["schemas"]
