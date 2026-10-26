/**
 * Manejo y actualización de la interfaz de usuario (DOM).
 * Centraliza los selectores y las funciones de renderizado visual.
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

  // Tarjeta superior: Especialista de género
  resAGender: document.getElementById("res-a-gender"),
  resALat: document.getElementById("res-a-lat"),

  // Tarjeta superior: Especialista de edad
  resBAge: document.getElementById("res-b-age"),
  resBLat: document.getElementById("res-b-lat"),

  // Tarjeta superior: Red multitarea unificada
  resCGender: document.getElementById("res-c-gender"),
  resCAge: document.getElementById("res-c-age"),
  resCLat: document.getElementById("res-c-lat"),
  resCSavings: document.getElementById("res-c-savings"),
  resCombinedLat: document.getElementById("res-combined-lat"),

  // Resumen comparativo inferior: Modelo de género
  metricGenPred: document.getElementById("metric-gen-pred"),
  metricGenLat: document.getElementById("metric-gen-lat"),
  metricGenSize: document.getElementById("metric-gen-size"),
  metricGenParams: document.getElementById("metric-gen-params"),

  // Resumen comparativo inferior: Modelo de edad
  metricAgePred: document.getElementById("metric-age-pred"),
  metricAgeLat: document.getElementById("metric-age-lat"),
  metricAgeSize: document.getElementById("metric-age-size"),
  metricAgeParams: document.getElementById("metric-age-params"),

  // Resumen comparativo inferior: Modelo multitarea
  metricMultiGender: document.getElementById("metric-multi-gender"),
  metricMultiAge: document.getElementById("metric-multi-age"),
  metricMultiLat: document.getElementById("metric-multi-lat"),
  metricMultiSize: document.getElementById("metric-multi-size"),
  metricMultiParams: document.getElementById("metric-multi-params"),
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
      ? "sample-btn h-7 sm:h-8 px-2 sm:px-2.5 py-1 rounded-lg text-xs font-mono-tag bg-primary/10 text-primary font-medium border border-primary/20 text-center flex items-center justify-center whitespace-nowrap"
      : "sample-btn h-7 sm:h-8 px-2 sm:px-2.5 py-1 rounded-lg text-xs font-mono-tag bg-surface-container text-on-surface-variant hover:bg-surface-container-high text-center flex items-center justify-center whitespace-nowrap";
  });
}

/**
 * Coloca los indicadores en estado de carga mientras se ejecuta la inferencia.
 */
export function setLoadingState() {
  // Tarjetas de predicción superior
  if (dom.resAGender) dom.resAGender.textContent = "Calculando...";
  if (dom.resALat) dom.resALat.textContent = "—";
  if (dom.resBAge) dom.resBAge.textContent = "Calculando...";
  if (dom.resBLat) dom.resBLat.textContent = "—";
  if (dom.resCGender) dom.resCGender.textContent = "Calculando...";
  if (dom.resCAge) dom.resCAge.textContent = "Calculando...";
  if (dom.resCLat) dom.resCLat.textContent = "—";
  if (dom.resCSavings) dom.resCSavings.textContent = "Midiendo ahorro...";
  if (dom.resCombinedLat) dom.resCombinedLat.textContent = "Calculando...";

  // Resumen corporativo inferior
  if (dom.metricGenPred) dom.metricGenPred.textContent = "Calculando...";
  if (dom.metricGenLat) dom.metricGenLat.textContent = "—";
  if (dom.metricAgePred) dom.metricAgePred.textContent = "Calculando...";
  if (dom.metricAgeLat) dom.metricAgeLat.textContent = "—";
  if (dom.metricMultiGender) dom.metricMultiGender.textContent = "Calculando...";
  if (dom.metricMultiAge) dom.metricMultiAge.textContent = "Calculando...";
  if (dom.metricMultiLat) dom.metricMultiLat.textContent = "—";
}

/**
 * Actualiza todos los valores de las tarjetas y tabla comparativa con la respuesta de la API.
 * @param {Object} data Respuesta devuelta por /api/predict
 */
export function renderModelOutputs(data) {
  if (!data || !data.success) return;

  // 1. Especialista de género
  if (dom.resAGender) dom.resAGender.textContent = data.specialist_gender.display;
  if (dom.resALat) dom.resALat.textContent = `${data.specialist_gender.latency_ms} ms`;
  if (dom.metricGenPred) dom.metricGenPred.textContent = data.specialist_gender.display;
  if (dom.metricGenLat) dom.metricGenLat.textContent = `${data.specialist_gender.latency_ms} ms`;

  // 2. Especialista de edad
  if (dom.resBAge) dom.resBAge.textContent = data.specialist_age.display;
  if (dom.resBLat) dom.resBLat.textContent = `${data.specialist_age.latency_ms} ms`;
  if (dom.metricAgePred) dom.metricAgePred.textContent = data.specialist_age.display;
  if (dom.metricAgeLat) dom.metricAgeLat.textContent = `${data.specialist_age.latency_ms} ms`;

  // 3. Red multitarea unificada
  if (dom.resCGender) dom.resCGender.textContent = data.multitask.gender_display;
  if (dom.resCAge) dom.resCAge.textContent = data.multitask.age_display;
  if (dom.resCLat) dom.resCLat.textContent = `${data.multitask.latency_ms} ms`;
  if (dom.metricMultiGender) dom.metricMultiGender.textContent = data.multitask.gender_display;
  if (dom.metricMultiAge) dom.metricMultiAge.textContent = data.multitask.age_display;
  if (dom.metricMultiLat) dom.metricMultiLat.textContent = `${data.multitask.latency_ms} ms`;

  // 4. Ahorro de latencia y latencia combinada
  if (dom.resCSavings) {
    const sign = data.multitask.savings_pct >= 0 ? "−" : "+";
    dom.resCSavings.textContent = `${sign}${Math.abs(data.multitask.savings_pct)}% vs serie`;
  }
  if (dom.resCombinedLat) {
    dom.resCombinedLat.textContent = `${data.combined_specialist_latency_ms} ms combinados`;
  }

  // 5. Dimensiones reales informadas por el backend
  if (data.image_dimensions && dom.dimensions) {
    dom.dimensions.textContent = data.image_dimensions;
  }

  // 6. Información técnica de tamaño y parámetros
  if (data.models_info) {
    if (dom.metricGenSize && data.models_info.gender) dom.metricGenSize.textContent = data.models_info.gender.size_str;
    if (dom.metricGenParams && data.models_info.gender) dom.metricGenParams.textContent = data.models_info.gender.params_str;
    if (dom.metricAgeSize && data.models_info.age) dom.metricAgeSize.textContent = data.models_info.age.size_str;
    if (dom.metricAgeParams && data.models_info.age) dom.metricAgeParams.textContent = data.models_info.age.params_str;
    if (dom.metricMultiSize && data.models_info.multitask) dom.metricMultiSize.textContent = data.models_info.multitask.size_str;
    if (dom.metricMultiParams && data.models_info.multitask) dom.metricMultiParams.textContent = data.models_info.multitask.params_str;
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
      ? "Recargando checkpoints (.pth)..."
      : "Modelos cargados automáticamente en memoria";
  }
}
