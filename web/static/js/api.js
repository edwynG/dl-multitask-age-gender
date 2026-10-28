/**
 * Cliente de comunicación con la API REST del backend Flask.
 * Maneja peticiones asíncronas para inferencia y estado de los modelos.
 */
export const ApiService = {
  /**
   * Envía una imagen (FormData o base64) al backend para inferencia facial.
   * @param {string|FormData} payload
   * @returns {Promise<Object|null>}
   */
  async predict(payload) {
    const isFormData = payload instanceof FormData;
    const options = isFormData
      ? { method: "POST", body: payload }
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image: payload }),
        };

    try {
      const response = await fetch("/api/predict/gender", options);
      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        console.warn("Error en /api/predict/gender:", response.status, err.error || "");
        return null;
      }
      return await response.json();
    } catch (err) {
      console.warn("Servidor Flask local no alcanzable en /api/predict/gender:", err.message);
      return null;
    }
  },

  /**
   * Solicita al backend recargar los checkpoints .pth desde la carpeta models/.
   * @returns {Promise<Object|null>}
   */
  async reloadModels() {
    try {
      const response = await fetch("/api/reload_models", { method: "POST" });
      if (!response.ok) return null;
      return await response.json();
    } catch (err) {
      console.warn("Error al recargar modelos:", err.message);
      return null;
    }
  },

  /**
   * Consulta el estado actual de los modelos y del dispositivo de inferencia.
   * @returns {Promise<Object|null>}
   */
  async getStatus() {
    try {
      const response = await fetch("/api/health");
      if (!response.ok) return null;
      return await response.json();
    } catch (err) {
      return null;
    }
  },
};
