import os
import torch
from dotenv import find_dotenv, load_dotenv

# Carga variables de entorno desde .env si existe
load_dotenv(find_dotenv())

# Identificador del modelo afinado en el Hub de Hugging Face
MODEL_ID = os.getenv("MODEL_ID", "miguelciav/vit-base-utkface-gender")

# Limitación defensiva de hilos en CPU para despliegues en la nube (Render / 512 MB RAM)
if os.environ.get("RENDER"):
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

# Selección automática del dispositivo de aceleración
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Métricas experimentales obtenidas en el notebook sobre el conjunto de prueba (UTKFace)
BENCHMARK_REGIMES = [
    {
        "id": "01",
        "nombre": "Baseline desde cero",
        "descripcion": "ViT-Tiny con pesos aleatorios sin preentrenar",
        "params": "1.97M",
        "exactitud_val": "69.19%",
        "exactitud_test": "67.04%",
        "tiempo": "284 s",
        "es_optimo": False,
    },
    {
        "id": "02",
        "nombre": "Base sin adaptar",
        "descripcion": "ViT-Base con cabeza lineal aleatoria sin ajustar",
        "params": "0",
        "exactitud_val": "64.52%",
        "exactitud_test": "65.41%",
        "tiempo": "23 s",
        "es_optimo": False,
    },
    {
        "id": "03",
        "nombre": "Feature extraction",
        "descripcion": "Backbone congelado, solo se entrena la cabeza clasificadora",
        "params": "1.5K",
        "exactitud_val": "94.37%",
        "exactitud_test": "93.56%",
        "tiempo": "243 s",
        "es_optimo": False,
    },
    {
        "id": "04",
        "nombre": "Fine-tuning completo",
        "descripcion": "Todas las capas optimizadas (modelo publicado en el Hub)",
        "params": "85.8M",
        "exactitud_val": "95.07%",
        "exactitud_test": "94.33%",
        "tiempo": "395 s",
        "es_optimo": True,
    },
    {
        "id": "05",
        "nombre": "Descongelado selectivo",
        "descripcion": "Últimos 2 bloques de atención y cabeza entrenados",
        "params": "14.18M",
        "exactitud_val": "94.78%",
        "exactitud_test": "93.78%",
        "tiempo": "273 s",
        "es_optimo": False,
    },
]

BENCHMARK_METRICS = {
    "gender": {
        "accuracy": "94.33%",
        "model": MODEL_ID,
    }
}
