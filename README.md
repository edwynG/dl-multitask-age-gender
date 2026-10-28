# Clasificación de género facial mediante Vision Transformer y transferencia de aprendizaje

**Redes neuronales profundas - Universidad Central de Venezuela (UCV) - Semestre I-2026**  
Profesor: Fernando Crema García

Repositorio del proyecto final para la adaptación de un modelo preentrenado a la tarea de clasificación de género en rostros a partir del catálogo UTKFace.

## Resumen del proyecto

Este proyecto evalúa cómo aprovechar modelos modernos de visión artificial para clasificar automáticamente el género de una persona a partir de fotos de rostros en situaciones reales, comparando el costo y la efectividad de distintas estrategias para la toma de decisiones.

### Datos utilizados y distribución

Trabajamos con 18.000 imágenes del catálogo público UTKFace con una distribución equilibrada de 52% hombres y 48% mujeres. Para garantizar que los resultados sean confiables y medibles en situaciones reales, el catálogo se dividió en tres grupos independientes: 12.600 fotografías para entrenar el sistema, 2.700 para supervisar los avances y 2.700 para una prueba final con imágenes que el sistema nunca había visto. Todas las fotos se estandarizaron a 224 píxeles y se aplicaron pequeñas variaciones como giros suaves e iluminación distinta para simular fotos tomadas en el día a día.

### Rendimiento y comparación de estrategias

En lugar de crear un sistema desde cero, evaluamos si convenía transferir el conocimiento de un modelo de visión ya existente llamado Vision Transformer. Los resultados demostraron una clara diferencia de valor. Entrenar un modelo desde la nada sin experiencia previa requirió mucho esfuerzo computacional y apenas alcanzó un 67% de acierto, lo cual es insuficiente para un entorno de negocio. Por el contrario, tomar un modelo ya entrenado y ajustar únicamente su decisión final elevó el acierto a un 93,5% en apenas 4 minutos de cómputo, demostrando una alta rentabilidad entre tiempo invertido y precisión. La mejor estrategia fue permitir que todo el modelo adaptara su atención a los rasgos de rostros humanos, logrando el primer lugar con un 94,3% de acierto general y manteniendo una precisión equilibrada tanto en hombres con 95,3% como en mujeres con 93,7%, lo que confirma que el sistema no tiene sesgos marcados hacia ningún grupo.

### Auditoría de fallas y calidad de la información

Al revisar con lupa los casos donde el sistema se equivocó con alta seguridad, encontramos que más del 60% de los fallos no fueron errores del modelo, sino equivocaciones de origen en el catálogo de datos con etiquetas invertidas de fábrica, sumado a fotos de recién nacidos cuyos rasgos faciales aún no presentan diferenciación visual clara.

### Recomendaciones estratégicas

Para una siguiente fase de inversión recomendamos implementar filtros automáticos de control de calidad para limpiar las etiquetas inconsistentes en la base de datos antes de entrenar, evaluar modelos diseñados exclusivamente para rostros y monitorear la precisión dividida por rangos de edad y etnias para asegurar un desempeño homogéneo en todos los perfiles de usuario.

### Disponibilidad del producto

El modelo ganador se encuentra publicado y disponible en la plataforma Hugging Face con el identificador miguelciav/vit-base-utkface-gender. Además se integró en una aplicación web interactiva que permite a directores y analistas probar el reconocimiento subiendo fotos o evaluando la cámara web en tiempo real.

## Enlaces del proyecto

* **Presentación del proyecto (PDF):** [docs/presentation/presentation.pdf](docs/presentation/presentation.pdf)
* **Notebook principal (Clasificación de género con Vision Transformer):** [notebook/vit_gender_transfer_learning.ipynb](notebook/vit_gender_transfer_learning.ipynb)
* **Notebook extra (Estimación conjunta de edad y género con MTL):** [notebook/extra/multitask_learning_age_gender.ipynb](notebook/extra/multitask_learning_age_gender.ipynb)
* **Modelo afinado en el Hub:** [miguelciav/vit-base-utkface-gender](https://huggingface.co/miguelciav/vit-base-utkface-gender)
* **Modelo base preentrenado:** [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)
* **Dataset en el Hub:** [nu-delta/utkface](https://huggingface.co/datasets/nu-delta/utkface)
* **Repositorio del proyecto:** [edwynG/dl-vit-gender-classification](https://github.com/edwynG/dl-vit-gender-classification)
* **Repositorio plantilla original:** [ucvia/dl26-proyecto](https://github.com/ucvia/dl26-proyecto)



## Estructura del repositorio

```text
dl-vit-gender-classification/
├── docs/
│   └── presentation/                        # Presentación oficial de defensa
│       ├── presentation.pdf                 #   Documento de diapositivas compilado
│       ├── presentation.tex                 #   Código fuente modular y configurable
│       └── figures/                         #   Gráficos del dataset, curvas, matriz y logos
├── notebook/
│   ├── vit_gender_transfer_learning.ipynb   # Experimentos principales en Google Colab
│   └── extra/                               # Investigación complementaria
│       └── multitask_learning_age_gender.ipynb #   Cuaderno de investigación extra (MTL)
├── api/                                     # Servicio de predicción que carga el modelo
│   ├── config.py                            #   Lectura de variables de configuración
│   ├── inference.py                         #   Inferencia con el modelo en memoria
│   ├── routes.py                            #   Rutas del servicio web
│   └── tests/test_api.py                    #   Pruebas automatizadas
├── web/                                     # Interfaz web interactiva
│   ├── routes.py                            #   Ruta principal que sirve la página
│   ├── templates/                           #   Plantillas visuales
│   └── static/                              #   Hojas de estilo y código de interacción
├── .env.example                             # Ejemplo de variables de configuración
├── requirements.txt                         # Librerías necesarias para ejecutar el proyecto
├── app.py                                   # Archivo principal que inicia todo el sistema
└── README.md                                # Documentación general del proyecto
```

> **Nota:** Ningún archivo de pesos (`.pth` o `.safetensors`) viaja en el repositorio. El backend descarga y cachea el modelo directamente desde Hugging Face Hub al iniciar.


## Regímenes experimentales comparados

En el notebook principal ([`notebook/vit_gender_transfer_learning.ipynb`](notebook/vit_gender_transfer_learning.ipynb)) se compararon los siguientes regímenes sobre particiones idénticas con semilla fija:

1. **Baseline desde cero:** ViT con inicialización aleatoria de pesos, sin preentrenamiento.
2. **Modelo base sin adaptar:** ViT preentrenado evaluado con cabeza lineal aleatoria sin ajustar.
3. **Feature extraction:** backbone congelado (`requires_grad = False`), optimizando únicamente la cabeza clasificadora.
4. **Fine-tuning completo:** todas las capas descongeladas con tasa de aprendizaje baja ($2\times 10^{-5}$).
5. **Experimento adicional (descongelado selectivo):** descongelado progresivo de las últimas capas del encoder de atención, evaluando la eficiencia de parámetros.


## Investigación exploratoria complementaria (Multi-Task Learning)

Como parte de una investigación complementaria y exploratoria de interés académico, se desarrolló un cuaderno adicional ubicado en [`notebook/extra/multitask_learning_age_gender.ipynb`](notebook/extra/multitask_learning_age_gender.ipynb) donde se evaluó el enfoque de **Aprendizaje Multitarea (Multi-Task Learning - MTL)** utilizando redes neuronales convolucionales sobre el mismo dataset UTKFace:

* **Objetivo:** Predecir simultáneamente la **edad** (regresión continua bajo pérdida Huber / Smooth $L_1$) y el **género** (clasificación binaria con Binary Cross-Entropy) a partir de un único rostro.
* **Tronco compartido (*Hard Parameter Sharing*):** Se comparó el desempeño de dos modelos especialistas independientes (uno por tarea) frente a una red con backbone convolucional compartido bifurcado en dos cabezas densas diferenciadas.
* **Resultados destacados:**
  * **Eficiencia de parámetros:** El modelo multitarea logró un ahorro del **48,6% en cantidad de parámetros** (1,2 millones frente a los 2,4 millones sumados de ambos especialistas) y resolvió ambas predicciones en una única pasada hacia adelante (*forward pass*).
  * **Transferencia inductiva positiva:** En clasificación de género, la red multitarea superó al especialista individual (**89,45% vs 88,72% de exactitud**, y **89,42% vs 87,88% en F1-score**), corroborando en la práctica los principios teóricos de Caruana (1997) donde las representaciones compartidas entre tareas correlacionadas actúan como un regularizador implícito.



## Extensión de la integración

Como extensión propia para la nota de integración (INT) y el factor $\alpha$, se implementó:
* **Cliente web interactivo con captura en vivo por cámara web y selector de muestras:** la interfaz permite evaluar el modelo en tiempo real mediante imágenes locales, muestras fijas del conjunto de prueba o captura directa desde la cámara web del usuario, comunicándose de forma asíncrona con el endpoint `/api/predict/gender` y midiendo la latencia de inferencia en milisegundos.
* **Diagnóstico y health check (`/api/health`):** endpoint que reporta el estado de carga del modelo en memoria, el dispositivo en uso (CPU/CUDA) y los segundos requeridos para la descarga inicial.


## Cómo correrlo

Requiere Python 3.11 o 3.12.

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Configurar variables de entorno
cp .env.example .env

# 3. Correr pruebas automatizadas
pytest api/tests -q

# 4. Iniciar la aplicación en desarrollo
python app.py

# Opción en producción con Gunicorn
gunicorn app:app --workers 1 --threads 2 --timeout 120
```

* **Cliente web:** [http://localhost:5000/](http://localhost:5000/)
* **Health check del API:** [http://localhost:5000/api/health](http://localhost:5000/api/health)
* **Endpoint de inferencia de género:** `POST http://localhost:5000/api/predict/gender`


## Declaración de uso de IA generativa

En cumplimiento con el reglamento del curso sobre el uso de herramientas de asistencia:

* **Código de integración (API REST y cliente web):** se utilizó asistencia de modelos de lenguaje (Claude / Gemini) como apoyo para la refactorización modular de Flask en Blueprints (`api/` y `web/`), la conexión del pipeline de Hugging Face con caché en memoria (`api/inference.py`), la llamada asíncrona con `fetch` en `api.js` y el diseño de los tests simulados con `pytest`. El código generado fue auditado, comentado y verificado manualmente mediante pruebas unitarias.
* **Análisis de datos y discusión del notebook:** el análisis exploratorio de datos, la configuración de hiperparámetros, la comparación de regímenes y la redacción de la discusión de resultados y limitaciones son de **redacción y autoría propia del equipo**.
