"""Servicio de carga e inferencia de modelos desde Hugging Face Hub.

El modelo se descarga y carga una sola vez (la primera vez que se solicita)
y queda cacheado en memoria para las siguientes peticiones.
"""

import io
import os
import time
import base64
from PIL import Image

from api.config import MODEL_ID, DEVICE

class ModelInferenceService:
    """Gestiona la carga desde el Hub de Hugging Face y la inferencia."""

    def __init__(self):
        self.model_id = MODEL_ID
        self.device = DEVICE
        self.clf = None
        self.status = {
            "ok": True,
            "cargado": False,
            "modelo": self.model_id,
            "dispositivo": str(self.device),
            "detalle": "",
            "segundos_carga": None,
        }
        # Intentamos precargar el modelo al instanciar
        self.precargar()

    def precargar(self):
        """Intenta precargar el pipeline de transformers."""
        try:
            self.get_pipeline()
        except Exception as e:
            self.status["cargado"] = False
            self.status["detalle"] = f"{type(e).__name__}: {e}"

    def get_pipeline(self):
        """Carga el pipeline de Hugging Face para MODEL_ID. Queda en memoria tras la primera carga."""
        if self.clf is not None:
            return self.clf

        from transformers import pipeline

        dispositivo_idx = 0 if self.device.type == "cuda" else -1
        t0 = time.time()
        self.clf = pipeline(
            "image-classification",
            model=self.model_id,
            device=dispositivo_idx,
            top_k=None,
        )
        t_total = round(time.time() - t0, 2)
        self.status.update({
            "cargado": True,
            "detalle": "",
            "segundos_carga": t_total,
            "dispositivo": str(self.device),
        })
        return self.clf

    def _parse_image(self, image_input) -> tuple[Image.Image, tuple[int, int]]:
        """Convierte una entrada (archivo binario, base64 o ruta) a una imagen PIL RGB."""
        if hasattr(image_input, "read"):
            # Objeto tipo archivo (UploadFile / FileStorage)
            content = image_input.read()
            pil_img = Image.open(io.BytesIO(content)).convert("RGB")
        elif isinstance(image_input, (bytes, bytearray)):
            pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, str):
            if os.path.exists(image_input):
                pil_img = Image.open(image_input).convert("RGB")
            else:
                if image_input.startswith("data:image"):
                    image_input = image_input.split(",")[1]
                image_bytes = base64.b64decode(image_input)
                pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise ValueError("Formato de imagen no soportado")

        dimensions = pil_img.size  # (ancho, alto)
        return pil_img, dimensions

    def predict(self, image_input) -> dict:
        """Ejecuta inferencia de género sobre una imagen usando el Vision Transformer."""
        pil_img, (w, h) = self._parse_image(image_input)

        # Inferencia con el pipeline de Hugging Face
        clf = self.get_pipeline()
        t_infer_start = time.time()
        salidas = clf(pil_img)
        latency_ms = round((time.time() - t_infer_start) * 1000, 1)

        # Normalizamos la salida del pipeline
        if salidas and isinstance(salidas[0], list):
            salidas = salidas[0]
        salidas = sorted(salidas, key=lambda r: r["score"], reverse=True)

        top_label = salidas[0]["label"] if salidas else "Desconocido"
        top_prob = round(salidas[0]["score"] * 100, 1) if salidas else 0.0

        label_map = {"Female": "Femenino", "Male": "Masculino"}
        label_es = label_map.get(top_label, top_label)

        # Mapeo limpio de probabilidades por clase
        probabilidades = {}
        prob_female = 0.0
        prob_male = 0.0
        for r in salidas:
            lbl = r["label"]
            score = round(float(r["score"]), 4)
            score_pct = round(score * 100, 1)
            probabilidades[label_map.get(lbl, lbl)] = score_pct
            if lbl.lower() in ("female", "femenino", "1"):
                prob_female = score_pct
            elif lbl.lower() in ("male", "masculino", "0"):
                prob_male = score_pct

        return {
            "success": True,
            "tarea": "clasificacion_genero",
            "genero": label_es,
            "clase": label_es,
            "genero_raw": top_label,
            "confianza": top_prob,
            "prob_female": prob_female,
            "prob_male": prob_male,
            "probabilidades": probabilidades,
            "latencia_ms": latency_ms,
            "dimensiones": f"{w}×{h} px",
            "modelo": self.model_id,
            "dispositivo": str(self.device),
        }

# Instancia singleton del servicio de inferencia
inference_service = ModelInferenceService()
