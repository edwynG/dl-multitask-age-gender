import os
import sys

# agregamos la raíz del proyecto a sys.path para permitir ejecución directa
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import datetime
from flask import Flask, render_template, request, jsonify
from web.inference import inference_service
from web.config import DEVICE, BENCHMARK_METRICS

# inicializamos la aplicación flask con las rutas a static y templates
app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "static"),
)


@app.context_processor
def inject_globals():
    """Inyecta variables globales en todas las plantillas Jinja2."""
    return {
        "current_year": datetime.now().year,
        "benchmark_metrics": BENCHMARK_METRICS,
    }


@app.route("/")
def index():
    """Ruta principal que renderiza la interfaz gráfica del laboratorio."""
    status = inference_service.status
    models_info = getattr(inference_service, "models_info", None)
    return render_template(
        "index.html",
        device=str(DEVICE),
        status=status,
        models_info=models_info,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Endpoint de inferencia facial que recibe una imagen y retorna las predicciones de los tres modelos."""
    try:
        # revisamos si se envió un archivo binario mediante formulario
        if "file" in request.files:
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"success": False, "error": "No se seleccionó ningún archivo."}), 400
            result = inference_service.predict(file)
            return jsonify(result)

        # revisamos si se envió un json con imagen en formato base64
        data = request.get_json(silent=True)
        if data and "image" in data:
            result = inference_service.predict(data["image"])
            return jsonify(result)

        # si no se encontró imagen en la petición
        return jsonify({"success": False, "error": "No se proporcionó ninguna imagen válida."}), 400

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/reload_models", methods=["POST"])
def api_reload_models():
    """Endpoint que vuelve a buscar y cargar los checkpoints de pesos desde la carpeta models/."""
    try:
        updated_status = inference_service.load_models()
        return jsonify({
            "success": True,
            "status": updated_status,
            "message": "Checkpoints recargados correctamente en memoria."
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/status", methods=["GET"])
def api_status():
    """Endpoint informativo para consultar el estado actual del dispositivo y modelos cargados."""
    return jsonify({
        "device": str(DEVICE),
        "status": inference_service.status,
    })


if __name__ == "__main__":
    # soporte para puerto asignado por Render ($PORT) o 5000 por defecto en local
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
