/**
 * Orquestador principal del Laboratorio Facial Multitarea.
 * Arquitectura modular ES6: coordina UI, API, Cámara Web y Muestras de prueba.
 */
import { SAMPLES } from "./samples.js";
import { ApiService } from "./api.js";
import { WebcamManager } from "./webcam.js";
import {
  dom,
  setDimensions,
  highlightSampleButton,
  setLoadingState,
  renderModelOutputs,
  setReloadButtonState,
} from "./ui.js";

document.addEventListener("DOMContentLoaded", () => {
  const webcam = new WebcamManager(dom.video);
  let liveTimer = null;

  /**
   * Ejecuta la inferencia en el backend y actualiza la interfaz con los resultados.
   * @param {string|FormData} payload
   */
  async function executeInference(payload) {
    try {
      const result = await ApiService.predict(payload);
      if (result) {
        renderModelOutputs(result);
      }
    } catch (err) {
      console.warn("Error durante la inferencia facial:", err);
    }
  }

  /**
   * Carga una de las muestras predeterminadas del conjunto de prueba.
   * @param {string} key Identificador de la muestra ('1', '2', '3')
   */
  async function applySample(key) {
    const sample = SAMPLES[key];
    if (!sample) return;

    if (webcam.isActive) stopWebcam();

    // Actualiza la imagen en el visor
    if (dom.preview) {
      dom.preview.classList.remove("hidden");
      dom.preview.style.opacity = "0.4";
      setTimeout(() => {
        dom.preview.src = sample.img;
        dom.preview.style.opacity = "1";
        dom.preview.onload = () => setDimensions(dom.preview.naturalWidth, dom.preview.naturalHeight);
      }, 100);
    }
    if (dom.video) dom.video.classList.add("hidden");
    if (dom.groundTruth) dom.groundTruth.textContent = sample.truth;

    highlightSampleButton(key);
    setLoadingState();

    // Descarga el archivo de imagen para enviarlo tal como está al backend
    try {
      const imgResp = await fetch(sample.img);
      const blob = await imgResp.blob();
      const formData = new FormData();
      formData.append("file", blob, `sample_${key}.jpg`);
      await executeInference(formData);
    } catch (err) {
      console.warn("No se pudo cargar la imagen de muestra:", err);
    }
  }

  /**
   * Manejador de subida de fotos por parte del usuario.
   */
  function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    if (webcam.isActive) stopWebcam();

    // Previsualización local inmediata
    const reader = new FileReader();
    reader.onload = (e) => {
      if (dom.preview) {
        dom.preview.classList.remove("hidden");
        dom.preview.src = e.target.result;
        dom.preview.onload = () => setDimensions(dom.preview.naturalWidth, dom.preview.naturalHeight);
      }
      if (dom.video) dom.video.classList.add("hidden");
      if (dom.groundTruth) dom.groundTruth.textContent = "Muestra personalizada del usuario";
    };
    reader.readAsDataURL(file);

    highlightSampleButton(null);
    setLoadingState();

    // Envío del archivo original sin compresión al backend
    const formData = new FormData();
    formData.append("file", file);
    executeInference(formData);
  }

  /**
   * Detiene la cámara web y restablece los controles de la interfaz.
   */
  function stopWebcam() {
    webcam.stop();
    if (liveTimer) {
      clearInterval(liveTimer);
      liveTimer = null;
    }

    if (dom.preview) dom.preview.classList.remove("hidden");
    if (dom.video) dom.video.classList.add("hidden");
    if (dom.liveIndicator) dom.liveIndicator.classList.add("hidden");
    if (dom.btnSnapshot) dom.btnSnapshot.classList.add("hidden");
    if (dom.labelFileInput) {
      dom.labelFileInput.classList.remove("hidden", "sm:flex");
    }

    if (dom.btnWebcam) {
      dom.btnWebcam.className =
        "h-9 sm:h-10 px-3.5 sm:px-4 py-2 rounded-xl text-xs sm:text-sm font-medium transition-all flex items-center justify-center gap-2 bg-surface-container hover:bg-surface-container-high text-on-surface active:scale-95 shadow-sm";
    }
    if (dom.camStatus) dom.camStatus.textContent = "Activar cámara";
    if (dom.camIcon) dom.camIcon.src = "/static/icons/videocam.svg";
  }

  /**
   * Inicia la captura continua de la cámara web.
   */
  async function startWebcam() {
    try {
      await webcam.start();

      if (dom.video) {
        dom.video.classList.remove("hidden");
        dom.video.onloadedmetadata = () => setDimensions(dom.video.videoWidth, dom.video.videoHeight);
      }
      if (dom.preview) dom.preview.classList.add("hidden");
      if (dom.liveIndicator) dom.liveIndicator.classList.remove("hidden");
      if (dom.btnSnapshot) {
        dom.btnSnapshot.className =
          "h-9 sm:h-10 px-3.5 sm:px-4 py-2 rounded-xl text-xs sm:text-sm font-medium transition-all flex items-center justify-center gap-2 bg-primary hover:bg-primary/90 text-on-primary shadow-sm active:scale-95";
        dom.btnSnapshot.classList.remove("hidden");
      }
      if (dom.labelFileInput) {
        dom.labelFileInput.classList.add("hidden", "sm:flex");
      }

      if (dom.btnWebcam) {
        dom.btnWebcam.className =
          "h-9 sm:h-10 px-3.5 sm:px-4 py-2 rounded-xl text-xs sm:text-sm font-medium transition-all flex items-center justify-center gap-2 bg-surface-container-high hover:bg-surface-container-highest text-on-surface border border-outline-variant/60 active:scale-95 shadow-sm";
      }
      if (dom.camStatus) dom.camStatus.textContent = "Detener cámara";
      if (dom.camIcon) dom.camIcon.src = "/static/icons/videocam_off.svg";
      if (dom.groundTruth) dom.groundTruth.textContent = "Transmisión activa de cámara web";

      captureAndPredict();
      liveTimer = setInterval(captureAndPredict, 1500);
    } catch (err) {
      alert("No se pudo acceder a la cámara web. Verifica los permisos de tu navegador.");
      stopWebcam();
    }
  }

  /**
   * Captura un fotograma del video y lo envía para inferencia en tiempo real.
   */
  function captureAndPredict() {
    const dataUrl = webcam.captureFrame(0.85);
    if (dataUrl) executeInference(dataUrl);
  }

  /**
   * Captura una foto fija desde la cámara web y congela la imagen en el visor.
   */
  function captureSnapshot() {
    const dataUrl = webcam.captureFrame(0.9);
    if (!dataUrl) return;

    if (dom.preview) {
      dom.preview.src = dataUrl;
      dom.preview.classList.remove("hidden");
    }
    if (dom.video) dom.video.classList.add("hidden");
    if (dom.groundTruth) dom.groundTruth.textContent = "Foto capturada por cámara web";

    stopWebcam();
    highlightSampleButton(null);
    setLoadingState();
    executeInference(dataUrl);
  }

  /**
   * Solicita recargar los modelos en memoria desde los archivos .pth.
   */
  async function reloadModels() {
    setReloadButtonState(true);
    await ApiService.reloadModels();
    setTimeout(() => {
      setReloadButtonState(false);
    }, 700);
  }

  // --- Registro de eventos de la interfaz ---

  // Botones de muestras predeterminadas
  dom.sampleBtns.forEach((btn) => {
    btn.addEventListener("click", () => applySample(btn.getAttribute("data-sample")));
  });

  // Selector de archivos del usuario
  if (dom.fileInput) {
    dom.fileInput.addEventListener("change", handleFileUpload);
  }

  // Botón de activación / desactivación de cámara web
  if (dom.btnWebcam) {
    dom.btnWebcam.addEventListener("click", () => {
      if (webcam.isActive) stopWebcam();
      else startWebcam();
    });
  }

  // Botón de captura fija de la cámara web
  if (dom.btnSnapshot) {
    dom.btnSnapshot.addEventListener("click", captureSnapshot);
  }

  // Botón de recarga de modelos
  if (dom.btnReload) {
    dom.btnReload.addEventListener("click", reloadModels);
  }

  // Inicialización: cargar la Muestra 1 al arrancar la aplicación
  applySample("1");
});
