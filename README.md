# Clasificación de género facial con vision transformer - transfer learning y fine-tuning con Hugging Face

**Redes neuronales profundas - Universidad Central de Venezuela (UCV) - Semestre I-2026**  
Profesor: Fernando Crema García

Repositorio del proyecto final para la adaptación de un modelo preentrenado del Hub de Hugging Face a la tarea de clasificación de género en rostros a partir del conjunto de datos UTKFace.

---

## Enlaces del proyecto

* **Modelo afinado en el Hub:** [miguelciav/vit-base-utkface-gender](https://huggingface.co/miguelciav/vit-base-utkface-gender)
* **Modelo base preentrenado:** [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)
* **Dataset en el Hub:** [nu-delta/utkface](https://huggingface.co/datasets/nu-delta/utkface)
* **Repositorio del proyecto:** [edwynG/dl-vit-gender-classification](https://github.com/edwynG/dl-vit-gender-classification)
* **Repositorio plantilla original:** [ucvia/dl26-proyecto](https://github.com/ucvia/dl26-proyecto)

---

## Estructura del repositorio

```text
dl-vit-gender-classification/
├── notebook/
│   └── vit_gender_transfer_learning.ipynb   # Experimentos de transfer learning y fine-tuning (Colab)
├── api/                                     # API REST: carga el modelo desde el Hub y sirve predicciones
│   ├── __init__.py                          #   Exporta el Blueprint api_bp
│   ├── config.py                            #   Lectura de .env (MODEL_ID, DEVICE, DEBUG)
│   ├── inference.py                         #   Carga única con transformers.pipeline en memoria e inferencia
│   ├── routes.py                            #   GET /api/health - POST /api/predict/gender
│   └── tests/test_api.py                    #   Pruebas con el modelo simulado (mock)
├── web/                                     # Cliente web interactivo que consume el API
│   ├── __init__.py                          #   Exporta el Blueprint web_bp
│   ├── routes.py                            #   GET / (renderiza la interfaz gráfica)
│   ├── templates/                           #   Plantillas Jinja2 (index.html, componentes)
│   └── static/                              #   CSS, iconos y JavaScript modular (api.js, webcam.js)
├── .env.example                             # Variables de entorno (MODEL_ID, PORT, DEBUG)
├── requirements.txt                         # Dependencias exactas del proyecto
├── app.py                                   # Orquestador que inicia el API y el cliente web
└── README.md                                # Documentación del proyecto
```

> **Nota:** Ningún archivo de pesos (`.pth` o `.safetensors`) viaja en el repositorio. El backend descarga y cachea el modelo directamente desde Hugging Face Hub al iniciar.

---

## Regímenes experimentales comparados

En el notebook ([`notebook/vit_gender_transfer_learning.ipynb`](notebook/vit_gender_transfer_learning.ipynb)) se compararon los siguientes regímenes sobre particiones idénticas con semilla fija:

1. **Baseline desde cero:** ViT con inicialización aleatoria de pesos, sin preentrenamiento.
2. **Modelo base sin adaptar:** ViT preentrenado evaluado con cabeza lineal aleatoria sin ajustar.
3. **Feature extraction:** backbone congelado (`requires_grad = False`), optimizando únicamente la cabeza clasificadora.
4. **Fine-tuning completo:** todas las capas descongeladas con tasa de aprendizaje baja ($2\times 10^{-5}$).
5. **Experimento adicional (descongelado selectivo):** descongelado progresivo de las últimas capas del encoder de atención, evaluando la eficiencia de parámetros.

---

## Extensión de la integración

Como extensión propia para la nota de integración (INT) y el factor $\alpha$, se implementó:
* **Cliente web interactivo con captura en vivo por cámara web y selector de muestras:** la interfaz permite evaluar el modelo en tiempo real mediante imágenes locales, muestras fijas del conjunto de prueba o captura directa desde la cámara web del usuario, comunicándose de forma asíncrona con el endpoint `/api/predict/gender` y midiendo la latencia de inferencia en milisegundos.
* **Diagnóstico y health check (`/api/health`):** endpoint que reporta el estado de carga del modelo en memoria, el dispositivo en uso (CPU/CUDA) y los segundos requeridos para la descarga inicial.

---

## Cómo correrlo

Requiere Python 3.11 o 3.12.

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Configurar variables de entorno
cp .env.example .env

# 3. Correr pruebas automatizadas del API (simulan el modelo, corren en segundos)
pytest api/tests -q

# 4. Iniciar la aplicación en desarrollo (API y cliente web unificados)
python app.py

# Opción en producción con Gunicorn (servidores en la nube como Render):
gunicorn app:app --workers 1 --threads 2 --timeout 120
```

* **Cliente web:** [http://localhost:5000/](http://localhost:5000/)
* **Health check del API:** [http://localhost:5000/api/health](http://localhost:5000/api/health)
* **Endpoint de inferencia de género:** `POST http://localhost:5000/api/predict/gender`

---

## Declaración de uso de IA generativa

En cumplimiento con el reglamento del curso sobre el uso de herramientas de asistencia:

* **Código de integración (API REST y cliente web):** se utilizó asistencia de modelos de lenguaje (Claude / Gemini) como apoyo para la refactorización modular de Flask en Blueprints (`api/` y `web/`), la conexión del pipeline de Hugging Face con caché en memoria (`api/inference.py`), la llamada asíncrona con `fetch` en `api.js` y el diseño de los tests simulados con `pytest`. El código generado fue auditado, comentado y verificado manualmente mediante pruebas unitarias.
* **Análisis de datos y discusión del notebook:** el análisis exploratorio de datos, la configuración de hiperparámetros, la comparación de regímenes y la redacción de la discusión de resultados y limitaciones son de **redacción y autoría propia del equipo**.
