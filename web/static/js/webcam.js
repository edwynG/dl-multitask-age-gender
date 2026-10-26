/**
 * Gestor modular de cámara web para captura en vivo e inferencia facial continua.
 * Soporta alternancia entre cámara frontal y trasera, y corrige el efecto espejo en dispositivos móviles.
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
    this.facingMode = "user"; // 'user' (frontal/selfie) o 'environment' (trasera)
    this.width = options.width || 480;
    this.height = options.height || 480;
    this.canvas = document.createElement("canvas");
    this.canvas.width = 200;
    this.canvas.height = 200;
    this.ctx = this.canvas.getContext("2d");
  }

  /**
   * Inicia la captura de video desde la cámara del usuario respetando el facingMode deseado.
   * @returns {Promise<MediaStream>}
   */
  async start() {
    if (this.isActive) return this.stream;

    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: this.width },
          height: { ideal: this.height },
          facingMode: { ideal: this.facingMode },
        },
        audio: false,
      });

      if (this.video) {
        this.video.srcObject = this.stream;
        this.applyMirrorStyle();
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
   * Aplica o remueve el efecto espejo según el tipo de cámara activa.
   * La cámara frontal ('user') se espeja para que la previsualización sea natural e intuitiva al moverse.
   * La cámara trasera ('environment') se muestra sin invertir para preservar la orientación real.
   */
  applyMirrorStyle() {
    if (!this.video) return;
    if (this.facingMode === "user") {
      this.video.style.transform = "scaleX(-1)";
      this.video.style.webkitTransform = "scaleX(-1)";
    } else {
      this.video.style.transform = "none";
      this.video.style.webkitTransform = "none";
    }
  }

  /**
   * Alterna entre cámara frontal y cámara trasera liberando primero el sensor anterior.
   * @returns {Promise<string>} Nuevo facingMode activo ('user' o 'environment')
   */
  async switchCamera() {
    const nextMode = this.facingMode === "user" ? "environment" : "user";
    this.facingMode = nextMode;

    if (this.isActive) {
      // Liberamos los tracks actuales antes de solicitar el sensor alternativo (crucial en iOS/Android)
      if (this.stream) {
        this.stream.getTracks().forEach((track) => track.stop());
        this.stream = null;
      }
      this.isActive = false;
      await this.start();
    }
    return this.facingMode;
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
      this.video.style.transform = "none";
      this.video.style.webkitTransform = "none";
    }
    this.isActive = false;
  }

  /**
   * Captura el fotograma actual de video como una imagen recortada cuadrada en formato base64.
   * Refleja horizontalmente si está en modo frontal para que la foto resultante coincida
   * con lo que el usuario observaba en el visor de la pantalla.
   * @param {number} quality calidad de compresión JPEG (0.0 a 1.0)
   * @returns {string|null} imagen DataURL en formato JPEG
   */
  captureFrame(quality = 0.85) {
    if (!this.isActive || !this.video) return null;

    const videoW = this.video.videoWidth;
    const videoH = this.video.videoHeight;
    if (!videoW || !videoH) return null;

    const minDim = Math.min(videoW, videoH);
    const startX = (videoW - minDim) / 2;
    const startY = (videoH - minDim) / 2;

    this.ctx.save();
    // En cámara frontal, reflejamos el dibujo en el canvas para mantener consistencia con el visor en pantalla
    if (this.facingMode === "user") {
      this.ctx.translate(this.canvas.width, 0);
      this.ctx.scale(-1, 1);
    }

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
    this.ctx.restore();

    return this.canvas.toDataURL("image/jpeg", quality);
  }
}
