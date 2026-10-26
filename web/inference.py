import os
import io
import time
import base64
from PIL import Image
import numpy as np
import torch

from web.config import (
    AGE_MODEL_PATH,
    GENDER_MODEL_PATH,
    MULTITASK_MODEL_PATH,
    IMAGE_SIZE,
    NORM_MEAN,
    NORM_STD,
    DEVICE,
)
from web.models import AgeEstimationModel, GenderClassificationModel, MultiTaskFaceModel


class ModelInferenceService:
    """Servicio encargado de gestionar el ciclo de vida de los modelos y ejecutar la inferencia."""

    def __init__(self):
        self.device = DEVICE
        self.model_age = None
        self.model_gender = None
        self.model_multitask = None
        self.status = {
            "age_loaded": False,
            "gender_loaded": False,
            "multitask_loaded": False,
            "device": str(self.device),
        }

        # tensores de normalización precalculados para evitar la pesada sobrecarga de torchvision
        self.norm_mean = torch.tensor(NORM_MEAN, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.norm_std = torch.tensor(NORM_STD, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

        # cargamos los modelos al inicializar el servicio
        self.load_models()

    def load_models(self):
        """Instancia y carga los pesos de los tres modelos en el dispositivo configurado."""
        # 1. modelo especialista en edad
        self.model_age = AgeEstimationModel().to(self.device)
        if os.path.exists(AGE_MODEL_PATH):
            self.model_age.load_weights(AGE_MODEL_PATH, device=self.device)
            self.status["age_loaded"] = True
        else:
            self.status["age_loaded"] = False
        self.model_age.eval()

        # 2. modelo especialista en género
        self.model_gender = GenderClassificationModel().to(self.device)
        if os.path.exists(GENDER_MODEL_PATH):
            self.model_gender.load_weights(GENDER_MODEL_PATH, device=self.device)
            self.status["gender_loaded"] = True
        else:
            self.status["gender_loaded"] = False
        self.model_gender.eval()

        # 3. modelo multitarea unificado
        self.model_multitask = MultiTaskFaceModel().to(self.device)
        if os.path.exists(MULTITASK_MODEL_PATH):
            self.model_multitask.load_weights(MULTITASK_MODEL_PATH, device=self.device)
            self.status["multitask_loaded"] = True
        else:
            self.status["multitask_loaded"] = False
        self.model_multitask.eval()

        # información de parámetros y peso en disco real de cada modelo
        self.models_info = {
            "gender": self._get_model_info(self.model_gender, GENDER_MODEL_PATH),
            "age": self._get_model_info(self.model_age, AGE_MODEL_PATH),
            "multitask": self._get_model_info(self.model_multitask, MULTITASK_MODEL_PATH),
        }
        self.status["models_info"] = self.models_info

        # warm-up exclusivo para gpu para evitar penalizaciones de cpu en entornos restringidos
        if self.device.type == "cuda":
            try:
                dummy = torch.zeros(1, 3, *IMAGE_SIZE, device=self.device)
                with torch.no_grad():
                    self.model_gender(dummy)
                    self.model_age(dummy)
                    self.model_multitask(dummy)
                    torch.cuda.synchronize(self.device)
            except Exception:
                pass

        return self.status

    @staticmethod
    def _get_model_info(model, path):
        """Calcula el número real de parámetros y el tamaño en disco del archivo .pth."""
        params = sum(p.numel() for p in model.parameters()) if model else 0
        params_str = f"{params:,}"
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            size_str = f"{size_mb:.1f} MB"
        else:
            size_str = "—"
        return {
            "params": params,
            "params_str": params_str,
            "size_str": size_str,
            "display": size_str,
        }

    def preprocess_image(self, image_input):
        """Convierte una entrada de imagen en un tensor normalizado y retorna también su tamaño original."""
        # si recibimos una ruta de archivo existente o un string base64
        if isinstance(image_input, str):
            if os.path.exists(image_input):
                pil_img = Image.open(image_input).convert("RGB")
            else:
                if image_input.startswith("data:image"):
                    image_input = image_input.split(",")[1]
                image_bytes = base64.b64decode(image_input)
                pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        elif isinstance(image_input, (bytes, bytearray)):
            pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            # asumimos que es un objeto de archivo de werkzeug/flask
            pil_img = Image.open(image_input).convert("RGB")

        orig_size = pil_img.size  # (ancho, alto) real de la imagen

        # redimensionamos e ingresamos el tensor normalizado [1, 3, H, W]
        resized = pil_img.resize(IMAGE_SIZE, Image.BILINEAR)
        arr = np.array(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        tensor = (tensor - self.norm_mean) / self.norm_std
        return tensor, orig_size

    def predict(self, image_input):
        """Ejecuta la inferencia en los tres modelos y mide la latencia de cada uno."""
        tensor, (img_w, img_h) = self.preprocess_image(image_input)

        with torch.no_grad():
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

            # inferencia del especialista de género
            t0 = time.perf_counter()
            prob_gender, pred_gen_class = self.model_gender.predict(tensor, device=self.device)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            t_gender_ms = round((time.perf_counter() - t0) * 1000.0, 1)

            # inferencia del especialista de edad
            t1 = time.perf_counter()
            pred_age = self.model_age.predict(tensor, device=self.device)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            t_age_ms = round((time.perf_counter() - t1) * 1000.0, 1)

            # inferencia de la red multitarea (un solo pase para ambas salidas)
            t2 = time.perf_counter()
            m_age, m_prob_gender, m_gen_class = self.model_multitask.predict(tensor, device=self.device)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            t_multi_ms = round((time.perf_counter() - t2) * 1000.0, 1)

        # interpretamos las etiquetas binarias (0: masculino, 1: femenino)
        gen_c = int(pred_gen_class[0])
        gen_p = float(prob_gender[0])
        gender_label = "Femenino" if gen_c == 1 else "Masculino"
        gender_conf = float(round(gen_p * 100.0 if gen_c == 1 else (1.0 - gen_p) * 100.0, 1))

        m_gen_c = int(m_gen_class[0])
        m_gen_p = float(m_prob_gender[0])
        m_gender_label = "Femenino" if m_gen_c == 1 else "Masculino"
        m_gender_conf = float(round(m_gen_p * 100.0 if m_gen_c == 1 else (1.0 - m_gen_p) * 100.0, 1))

        age_val = float(round(max(1.0, float(pred_age[0])), 1))
        m_age_val = float(round(max(1.0, float(m_age[0])), 1))
        age_unit = "año" if age_val == 1.0 else "años"
        m_age_unit = "año" if m_age_val == 1.0 else "años"

        # cálculo de latencia combinada y ahorro porcentual de tiempo
        t_combined_ms = float(round(t_gender_ms + t_age_ms, 1))
        savings_pct = float(round(((t_combined_ms - t_multi_ms) / max(t_combined_ms, 0.1)) * 100.0, 1)) if t_combined_ms > 0 else 0.0

        return {
            "success": True,
            "image_dimensions": f"{img_w}×{img_h} px",
            "models_info": getattr(self, "models_info", None),
            "specialist_gender": {
                "label": gender_label,
                "confidence": gender_conf,
                "display": f"{gender_label} ({gender_conf}%)",
                "latency_ms": t_gender_ms,
            },
            "specialist_age": {
                "age": age_val,
                "display": f"{age_val} {age_unit}",
                "latency_ms": t_age_ms,
            },
            "multitask": {
                "gender_label": m_gender_label,
                "gender_confidence": m_gender_conf,
                "gender_display": f"{m_gender_label} ({m_gender_conf}%)",
                "age": m_age_val,
                "age_display": f"{m_age_val} {m_age_unit}",
                "latency_ms": t_multi_ms,
                "savings_pct": savings_pct,
            },
            "combined_specialist_latency_ms": t_combined_ms,
        }


# instancia singleton para uso en toda la aplicación web
inference_service = ModelInferenceService()
