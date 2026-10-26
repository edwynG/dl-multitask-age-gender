# Aprendizaje multitarea para estimación de edad y género en rostros

**Redes neuronales profundas [UCV] - Semestre I-2026**

El reconocimiento automático de atributos faciales es uno de los problemas clásicos de la visión por computador con aplicaciones directas en biometría y seguridad. Este trabajo de investigación evalúa si una única red convolucional con representaciones compartidas (Multi-Task Learning) puede predecir de forma simultánea la edad y el género de un individuo con un desempeño comparable o superior al de dos modelos especializados e independientes, optimizando el consumo de parámetros y el tiempo de inferencia.

> [!NOTE]
> Este proyecto tiene propósitos estrictamente académicos y de investigación científica en deep learning para la Universidad Central de Venezuela. No persigue fines comerciales ni de explotación de producto.

## Dataset y formulación del problema

El estudio se apoya en el conjunto de datos **UTKFace**, compuesto por más de 20.000 imágenes faciales en entornos no controlados (diversidad de poses, iluminación y etnias). Los nombres de los archivos codifican los atributos bajo el formato `[edad]_[género]_[etnia]_[fecha].jpg`.

Las tareas a resolver son:
* **Estimación de edad:** Problema de regresión continua (rango de 0 a 116 años).
* **Clasificación de género:** Problema de clasificación binaria (0: Masculino, 1: Femenino).

> [!IMPORTANT]
> El dataset contiene imágenes con rostros e identidades de personas. Su uso en este repositorio se limita exclusivamente a fines experimentales y de investigación académica para analizar la capacidad de generalización y sesgos en redes neuronales.

## Metodología experimental (SEMMA)

El flujo de trabajo implementado en el notebook se estructura bajo las cinco fases de la metodología **SEMMA**:

* **Muestreo (Sample):** Descarga dinámica mediante `kagglehub`, descarte de imágenes mal etiquetadas y división estratificada conjunta (18 estratos edad-género) en entrenamiento (70%), validación (15%) y prueba (15%).
* **Exploración (Explore):** Auditoría de dimensiones (todas las imágenes a 200×200 píxeles, RGB), análisis estadístico de la distribución de edades y detección de sesgos demográficos (sobrerrepresentación del rango 20 a 35 años).
* **Modificación (Modify):** Normalización calculando la media y desviación estándar de los canales de píxeles sobre el conjunto de entrenamiento, junto con técnicas de aumentación de datos (*random horizontal flip*, rotación leve y variación de brillo y contraste).
* **Modelado (Model):** Diseño de un extractor de características convolucional base (*backbone*) y contraste de tres aproximaciones:
  1. Modelo especialista en edad (regresión optimizada con función de pérdida MSE o L1).
  2. Modelo especialista en género (clasificación binaria optimizada con entropía cruzada binaria).
  3. Modelo multitarea con tronco compartido y dos cabezas lineales diferenciadas, optimizado mediante una pérdida ponderada conjunta.
* **Evaluación (Assess):** Comparación cuantitativa final sobre el conjunto de prueba independiente usando métricas de regresión (MAE, RMSE) y clasificación (exactitud, F1-score), además de auditar el ahorro en parámetros y casos de error cualitativo.

Toda la deducción matemática, el código ejecutable y las conclusiones experimentales completas se encuentran detalladas dentro del propio notebook.

## Arquitectura del repositorio

```text
dl-multitask-age-gender/
├── multitask_learning_age_gender.ipynb   # Informe principal y código del experimento (autocontenido)
├── models/                               # Checkpoints y pesos guardados de los modelos (.pth)
├── web/                                  # Demo interactiva en Flask para comparar los modelos
│   ├── app.py                            # Servidor web ligero
│   ├── models.py                         # Definición de las arquitecturas para inferencia
│   └── templates/
│       └── index.html                    # Interfaz visual de prueba
├── docs/                                 # Clases, libros y material teórico de referencia
├── requirements.txt                      # Dependencias del proyecto (PyTorch con aceleración CUDA)
└── README.md                             # Descripción general del repositorio
```

## Ejecución del proyecto

### 1. Preparación del entorno virtual
Crear y activar el entorno virtual con Python 3.12:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instalar las dependencias:
```bash
pip install -r requirements.txt
```

> [!NOTE]
> **Optimización de dependencias (`requirements.txt`):**
> * **Para la aplicación web / despliegue en Render:** El archivo instala únicamente el conjunto mínimo necesario para inferencia (`torch`, `flask`, `gunicorn`, `pillow`, `numpy`). Las librerías pesadas de análisis y entrenamiento (`pandas`, `matplotlib`, `seaborn`, `jupyter`, `torchvision`) están comentadas para evitar sobrepasar los 512 MB de RAM de Render y acelerar los tiempos de despliegue. **Esto no afecta en absoluto la precisión ni el rendimiento de los modelos en producción.**
> * **Para reentrenar el notebook:** Si deseas reejecutar el pipeline completo de entrenamiento y EDA desde cero, descomenta e instala las dependencias de la sección inferior de `requirements.txt`.

---

### 2. Ejecutar la aplicación web interactiva en local
Para iniciar el servidor web de desarrollo y evaluar los modelos en tiempo real con muestras, fotos propias o cámara web:
```bash
python -m web.app
# O alternativamente:
python web/app.py
```
Luego abre tu navegador en:
```text
http://127.0.0.1:5000
```

---

### 3. Ejecutar el Notebook de investigación
Para explorar el análisis exploratorio de datos (EDA), deducción de funciones de pérdida y entrenamiento experimental:
1. Abrir `multitask_learning_age_gender.ipynb` en VS Code, JupyterLab o Google Colab.
2. Seleccionar el kernel del entorno virtual (`.venv`).
3. Ejecutar secuencialmente las celdas del informe.
