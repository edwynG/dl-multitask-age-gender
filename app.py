"""Punto de entrada principal del Laboratorio de Deep Learning.

Arquitectura desacoplada:
  - api_bp (/api): Backend REST para inferencia con modelos de Hugging Face Hub.
  - web_bp (/): Frontend interactivo que consume el API vía HTTP.

Ejecutar:
  python app.py
"""

import os
from datetime import datetime
from dotenv import find_dotenv, load_dotenv
from flask import Flask

from api import api_bp
from api.config import BENCHMARK_METRICS, BENCHMARK_REGIMES, MODEL_ID
from web import web_bp

# Carga variables de entorno desde .env
load_dotenv(find_dotenv())

# Lectura del modo de depuración desde el entorno
debug_mode = os.getenv("DEBUG", "").strip().lower() in ("true", "1", "yes")

# Inicialización de la aplicación Flask
app = Flask(
    __name__,
    template_folder="web/templates",
    static_folder="web/static",
)

# En modo depuración se habilita la recarga automática de plantillas y se desactiva la caché de estáticos
if debug_mode:
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


# Inyección de variables globales en las plantillas Jinja2
@app.context_processor
def inject_globals():
    return {
        "current_year": datetime.now().year,
        "benchmark_metrics": BENCHMARK_METRICS,
        "benchmark_regimes": BENCHMARK_REGIMES,
        "model_id": MODEL_ID,
    }


# Registro de Blueprints modulares
app.register_blueprint(api_bp, url_prefix="/api")
app.register_blueprint(web_bp, url_prefix="/")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=debug_mode)


