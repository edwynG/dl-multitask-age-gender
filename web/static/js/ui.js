/**
 * Manejo y actualización de la interfaz de usuario (DOM).
 * Especializado para el laboratorio de Clasificación de Género con Vision Transformer.
 */

// Cache central de selectores del DOM
export const dom = {
  // Visor y previsualización
  preview: document.getElementById("face-preview"),
  video: document.getElementById("webcam-video"),
  dimensions: document.getElementById("preview-dimensions"),
  groundTruth: document.getElementById("label-ground-truth"),
  liveIndicator: document.getElementById("live-indicator"),

  // Controles e interacción
  sampleBtns: document.querySelectorAll(".sample-btn"),
  fileInput: document.getElementById("file-input"),
  labelFileInput: document.getElementById("label-file-input"),
  btnWebcam: document.getElementById("btn-webcam"),
  camStatus: document.getElementById("cam-status"),
  camIcon: document.getElementById("cam-icon"),
  btnSnapshot: document.getElementById("btn-capture-snapshot"),
  btnFlipFloating: document.getElementById("btn-flip-floating"),
  btnReload: document.getElementById("btn-reload-models"),
  iconReload: document.getElementById("icon-reload"),
  statusText: document.getElementById("status-text"),
  statusPulse: document.getElementById("status-pulse"),
  footerYear: document.getElementById("footer-year"),

  // Resultados de predicción ViT
  resGender: document.getElementById("res-gender"),
  resConfidence: document.getElementById("res-confidence"),
  resLatency: document.getElementById("res-latency"),
  resDevice: document.getElementById("res-device"),

  // Barras de distribución de probabilidades
  probFemaleBar: document.getElementById("prob-female-bar"),
  probFemaleVal: document.getElementById("prob-female-val"),
  probMaleBar: document.getElementById("prob-male-bar"),
  probMaleVal: document.getElementById("prob-male-val"),
};

/**
 * Muestra las dimensiones nativas de la imagen analizada.
 * @param {number} w ancho en píxeles
 * @param {number} h alto en píxeles
 */
export function setDimensions(w, h) {
  if (dom.dimensions && w && h) {
    dom.dimensions.textContent = `${w}×${h} px`;
  }
}

/**
 * Destaca visualmente el botón de la muestra actualmente activa.
 * @param {string|null} activeKey Clave de la muestra seleccionada ('1', '2', '3') o null
 */
export function highlightSampleButton(activeKey) {
  dom.sampleBtns.forEach((btn) => {
    const isSelected = btn.getAttribute("data-sample") === activeKey;
    btn.className = isSelected
      ? "sample-btn h-8 px-2.5 py-1 rounded-lg text-xs font-mono-tag bg-primary/10 text-primary font-medium border border-primary/20 text-center flex items-center justify-center whitespace-nowrap shrink-0"
      : "sample-btn h-8 px-2.5 py-1 rounded-lg text-xs font-mono-tag bg-surface-container text-on-surface-variant hover:bg-surface-container-high text-center flex items-center justify-center whitespace-nowrap shrink-0";
  });
}

/**
 * Coloca los indicadores en estado de carga mientras se ejecuta la inferencia.
 */
export function setLoadingState() {
  if (dom.resGender) dom.resGender.textContent = "Calculando...";
  if (dom.resConfidence) dom.resConfidence.textContent = "—";
  if (dom.resLatency) dom.resLatency.textContent = "...";
}

/**
 * Actualiza los valores de la predicción de género y las barras de probabilidad.
 * @param {Object} data Respuesta devuelta por /api/predict
 */
export function renderModelOutputs(data) {
  if (!data || !data.success) return;

  // 1. Género predicho y confianza
  const genero = data.genero || data.clase || "Desconocido";
  const confianza = data.confianza ? `${data.confianza}%` : "—";

  if (dom.resGender) {
    dom.resGender.textContent = genero;
    // Color dinámico según la clase
    if (genero.toLowerCase().startsWith("fem")) {
      dom.resGender.className = "text-2xl sm:text-3xl font-bold text-rose-600 block mt-0.5";
    } else {
      dom.resGender.className = "text-2xl sm:text-3xl font-bold text-blue-600 block mt-0.5";
    }
  }

  if (dom.resConfidence) {
    dom.resConfidence.textContent = confianza;
  }

  if (dom.resLatency && data.latencia_ms !== undefined) {
    dom.resLatency.textContent = `${data.latencia_ms} ms`;
  }

  // 2. Barras de probabilidades
  const pFemale = data.prob_female !== undefined ? data.prob_female : (data.probabilidades?.Femenino || 0);
  const pMale = data.prob_male !== undefined ? data.prob_male : (data.probabilidades?.Masculino || 0);

  if (dom.probFemaleBar) {
    dom.probFemaleBar.style.width = `${pFemale}%`;
  }
  if (dom.probFemaleVal) {
    dom.probFemaleVal.textContent = `${pFemale}%`;
  }

  if (dom.probMaleBar) {
    dom.probMaleBar.style.width = `${pMale}%`;
  }
  if (dom.probMaleVal) {
    dom.probMaleVal.textContent = `${pMale}%`;
  }

  // 3. Dimensiones
  if (dom.dimensions && data.dimensiones) {
    dom.dimensions.textContent = data.dimensiones;
  }

  // 4. Dispositivo
  if (dom.resDevice && data.dispositivo) {
    dom.resDevice.textContent = data.dispositivo;
  }
}

/**
 * Actualiza el estado visual del botón de recarga de modelos.
 * @param {boolean} isLoading
 */
export function setReloadButtonState(isLoading) {
  if (!dom.btnReload) return;
  dom.btnReload.disabled = isLoading;
  if (dom.iconReload) {
    dom.iconReload.classList.toggle("animate-spin", isLoading);
  }
  if (dom.statusPulse) {
    dom.statusPulse.className = isLoading
      ? "w-2 h-2 rounded-full bg-amber-500 animate-pulse"
      : "w-2 h-2 rounded-full bg-emerald-600 animate-pulse";
  }
  if (dom.statusText) {
    dom.statusText.textContent = isLoading
      ? "Recargando modelo..."
      : "Modelo ViT en memoria";
  }
}
