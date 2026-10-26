/**
 * Gestor modular de cámara web para captura en vivo e inferencia facial continua.
 */
export class WebcamManager {
  /**
   * @param {HTMLVideoElement} videoElement
   * @param {Object} options
   */
  constructor(videoElement, options = {}) {
    this.video = videoElement;
    this.stream = null;
    this.isActive = false;
    this.width = options.width || 480;
    this.height = options.height || 480;
    this.canvas = document.createElement("canvas");
    this.canvas.width = 200;
    this.canvas.height = 200;
    this.ctx = this.canvas.getContext("2d");
  }

  /**
   * Inicia la captura de video desde la cámara del usuario.
   * @returns {Promise<MediaStream>}
   */
  async start() {
    if (this.isActive) return this.stream;

    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: this.width },
          height: { ideal: this.height },
          facingMode: "user",
        },
        audio: false,
      });

      if (this.video) {
        this.video.srcObject = this.stream;
        await this.video.play();
      }
      this.isActive = true;
      return this.stream;
    } catch (err) {
      console.error("Error al acceder a la cámara web:", err);
      throw new Error("No se pudo acceder a la cámara web. Verifica los permisos de tu navegador.");
    }
  }

  /**
   * Detiene el flujo de la cámara y libera los recursos del hardware.
   */
  stop() {
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }
    if (this.video) {
      this.video.srcObject = null;
    }
    this.isActive = false;
  }

  /**
   * Captura el fotograma actual de video como una imagen recortada cuadrada en formato base64.
   * @param {number} quality calidad de compresión JPEG (0.0 a 1.0)
   * @returns {string|null} imagen DataURL en formato JPEG
   */
  captureFrame(quality = 0.85) {
    if (!this.isActive || !this.video) return null;

    // Calculamos las coordenadas para un recorte cuadrado centrado sin distorsión
    const videoW = this.video.videoWidth;
    const videoH = this.video.videoHeight;
    const minDim = Math.min(videoW, videoH);
    const startX = (videoW - minDim) / 2;
    const startY = (videoH - minDim) / 2;

    this.ctx.drawImage(
      this.video,
      startX,
      startY,
      minDim,
      minDim,
      0,
      0,
      this.canvas.width,
      this.canvas.height
    );

    return this.canvas.toDataURL("image/jpeg", quality);
  }
}
