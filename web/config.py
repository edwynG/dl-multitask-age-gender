import os
import torch

# limitamos hilos en cpu exclusivamente en producción (Render) para evitar contención de hilos
if os.environ.get("RENDER"):
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

# directorio base del proyecto y subcarpetas
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")

# rutas a los archivos de pesos guardados (.pth)
AGE_MODEL_PATH = os.path.join(MODELS_DIR, "model_age_best.pth")
GENDER_MODEL_PATH = os.path.join(MODELS_DIR, "model_gender_best.pth")
MULTITASK_MODEL_PATH = os.path.join(MODELS_DIR, "model_multitask_best.pth")

# resolución de entrada esperada por el extractor de características
IMAGE_SIZE = (200, 200)

# medias y desviaciones estándar calculadas sobre el conjunto de entrenamiento de utkface
NORM_MEAN = [0.5966, 0.4566, 0.3907]
NORM_STD = [0.2591, 0.2315, 0.2270]

# selección automática del dispositivo de inferencia (gpu cuda si está disponible)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
