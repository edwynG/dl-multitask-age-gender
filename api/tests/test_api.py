"""Pruebas del API con el modelo simulado (mock).
Corren en segundos y no descargan nada de Hugging Face.

Ejecutar:
    pytest api/tests -q
"""

import os
import sys
import io
import pytest
from PIL import Image

# Permite importar desde la raíz del proyecto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app import app
from api.inference import inference_service


class PipelineFalso:
    """Simula la salida de un pipeline de Hugging Face transformers."""
    def __call__(self, imagen):
        return [
            {"label": "Femenino", "score": 0.88},
            {"label": "Masculino", "score": 0.12},
        ]


@pytest.fixture
def client(monkeypatch):
    """Fixture que reemplaza el pipeline real por el simulado."""
    monkeypatch.setattr(inference_service, "get_pipeline", lambda: PipelineFalso())
    inference_service.status.update(cargado=True, detalle="", dispositivo="cpu")
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_health_ok(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert data["cargado"] is True
    assert "modelo" in data


def test_predict_image_ok(client):
    # Generamos una imagen sintética en memoria
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color="blue").save(buf, format="JPEG")
    buf.seek(0)

    r = client.post(
        "/api/predict/gender",
        data={"file": (buf, "test_face.jpg")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 200
    data = r.get_json()
    assert data["success"] is True
    assert data["tarea"] == "clasificacion_genero"
    assert data["clase"] == "Femenino"
    assert "probabilidades" in data
    assert "Femenino" in data["probabilidades"]


def test_predict_sin_imagen(client):
    r = client.post("/api/predict/gender", data={})
    assert r.status_code == 400
    data = r.get_json()
    assert data["success"] is False
    assert data["error"]["codigo"] == "validacion"
