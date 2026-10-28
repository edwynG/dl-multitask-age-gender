from flask import Blueprint, render_template
from api.inference import inference_service
from api.config import DEVICE, MODEL_ID

web_bp = Blueprint(
    "web",
    __name__,
    template_folder="templates",
    static_folder="static",
)


@web_bp.route("/")
def index():
    """Renderiza la vista principal del laboratorio y cliente web."""
    status = inference_service.status
    models_info = getattr(inference_service, "models_info", None)
    return render_template(
        "index.html",
        device=str(DEVICE),
        status=status,
        models_info=models_info,
        model_id=MODEL_ID,
    )
