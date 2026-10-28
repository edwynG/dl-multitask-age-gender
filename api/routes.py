from flask import Blueprint, request, jsonify
from api.inference import inference_service
from api.config import DEVICE

api_bp = Blueprint("api", __name__)


def _error_response(status_code: int, codigo: str, mensaje: str):
    """Genera una respuesta de error con la estructura estándar exigida por la rúbrica."""
    return jsonify({
        "success": False,
        "error": {
            "codigo": codigo,
            "mensaje": mensaje,
        }
    }), status_code


@api_bp.route("/health", methods=["GET"])
def health():
    """Endpoint de diagnóstico y estado del servicio (health check)."""
    status = inference_service.status
    http_code = 200 if status.get("cargado") else 503
    return jsonify({
        **status,
        "device": str(DEVICE),
    }), http_code


@api_bp.route("/predict/gender", methods=["POST"])
def predict_gender():
    """Endpoint de inferencia facial que clasifica el género de una imagen (archivo o captura de cámara web)."""
    try:
        # 1. Verificamos que el modelo esté cargado o intentamos cargarlo
        if not inference_service.status.get("cargado"):
            inference_service.precargar()
            if not inference_service.status.get("cargado"):
                return _error_response(
                    503, "modelo", f"El modelo no está cargado: {inference_service.status.get('detalle', 'Error desconocido')}"
                )

        # 2. Extracción de la imagen (multipart file o json base64)
        image_input = None
        if "file" in request.files and request.files["file"].filename != "":
            image_input = request.files["file"]
        elif request.is_json:
            data = request.get_json(silent=True) or {}
            image_input = data.get("image")

        if not image_input:
            return _error_response(400, "validacion", "Debe proporcionar una imagen en 'file' (formulario) o 'image' (base64).")

        # 3. Inferencia
        result = inference_service.predict(image_input)
        return jsonify(result), 200

    except Exception as e:
        return _error_response(500, "inferencia", str(e))


@api_bp.route("/reload_models", methods=["POST"])
def reload_models():
    """Fuerza la recarga del pipeline de Hugging Face desde el Hub."""
    try:
        inference_service.clf = None
        inference_service.precargar()
        return jsonify({
            "success": inference_service.status["cargado"],
            "status": inference_service.status,
            "message": "Pipeline de Hugging Face recargado."
        })
    except Exception as e:
        return _error_response(500, "interno", str(e))
