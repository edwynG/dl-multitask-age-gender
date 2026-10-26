import os
import time
import numpy as np
import torch
import torch.nn as nn

# carpeta para almacenar los modelos entrenados
os.makedirs("models", exist_ok=True)


class MultiTaskLoss(nn.Module):
    """Función de costo combinada para entrenar edad y género a la vez."""

    def __init__(self, weight_age=1.0, weight_gender=1.0):
        super(MultiTaskLoss, self).__init__()
        # guardamos los pesos para balancear la contribución de cada tarea
        self.weight_age = weight_age
        self.weight_gender = weight_gender
        # pérdidas específicas: huber para regresión continua y bce para clasificación binaria
        self.loss_age = nn.SmoothL1Loss()
        self.loss_gender = nn.BCEWithLogitsLoss()

    def forward(self, pred_age, pred_gender, target_age, target_gender):
        """Calcula el costo total sumando las pérdidas individuales ponderadas."""
        # calculamos el error en edad y en género por separado
        l_age = self.loss_age(pred_age, target_age)
        l_gender = self.loss_gender(pred_gender, target_gender)
        # sumamos linealmente ambos costos con sus pesos
        l_total = (self.weight_age * l_age) + (self.weight_gender * l_gender)
        return l_total, l_age, l_gender


class ConvBlock(nn.Module):
    """Bloque convolucional compuesto por dos capas Conv2D con BatchNorm y ReLU, seguido de MaxPool2d."""

    def __init__(self, in_channels: int, out_channels: int):
        super(ConvBlock, self).__init__()
        # definimos la secuencia: conv -> bn -> relu -> conv -> bn -> relu -> maxpool
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pasa la imagen por el bloque convolucional reduciendo su tamaño a la mitad."""
        # aplicamos las transformaciones del bloque secuencial
        return self.block(x)


class FeatureExtractorBackbone(nn.Module):
    """Extractor convolucional base que transforma la imagen 200x200x3 en un vector de 256 características."""

    def __init__(self):
        super(FeatureExtractorBackbone, self).__init__()
        # apilamos 4 bloques convolucionales que aumentan los canales de 3 a 256
        self.features = nn.Sequential(
            ConvBlock(3, 32),    # 200x200 -> 100x100
            ConvBlock(32, 64),   # 100x100 -> 50x50
            ConvBlock(64, 128),  # 50x50   -> 25x25
            ConvBlock(128, 256), # 25x25   -> 12x12
        )
        # pooling promedio global para obtener un vector de salida independiente de la resolución
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extrae el vector plano de 256 características a partir de la imagen."""
        # pasamos por los bloques convolucionales y luego por el pooling global
        x = self.features(x)
        x = self.gap(x)
        # aplanamos manteniendo el tamaño del lote [B, 256]
        return torch.flatten(x, 1)


class BaseFaceModel(nn.Module):
    """Clase base con métodos integrados para configurar optimizadores, entrenar, guardar e inferir."""

    def __init__(self):
        super(BaseFaceModel, self).__init__()

    def count_parameters(self) -> int:
        """Cuenta la cantidad total de parámetros entrenables."""
        # sumamos la cantidad de parámetros que requieren gradiente
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def configure_optimizer(self, lr: float = 1e-3, weight_decay: float = 1e-4):
        """Configura el optimizador AdamW y el planificador ReduceLROnPlateau."""
        # creamos adamw con regularización l2 y el planificador por mesetas
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )
        return optimizer, scheduler

    def train_epoch(self, loader, optimizer, device):
        """Entrena el modelo durante una época completa."""
        # ponemos el modelo en modo entrenamiento
        self.train()
        total_loss = 0.0
        for images, targets in loader:
            images = images.to(device)
            # reiniciamos gradientes del optimizador
            optimizer.zero_grad()
            outputs = self(images)
            # calculamos el costo y retropropagamos los gradientes
            loss = self.compute_loss(outputs, targets, device)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        # retornamos la pérdida promedio por muestra
        return total_loss / len(loader.dataset)

    def validate_epoch(self, loader, device):
        """Evalúa el modelo en validación sin calcular gradientes."""
        # ponemos el modelo en modo evaluación
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(device)
                outputs = self(images)
                # calculamos el error en validación
                loss = self.compute_loss(outputs, targets, device)
                total_loss += loss.item() * images.size(0)
        # retornamos la pérdida promedio en validación
        return total_loss / len(loader.dataset)

    def fit(self, train_loader, val_loader, epochs=15, lr=1e-3, device="cuda", save_path=None):
        """Ejecuta el entrenamiento completo por épocas y guarda el mejor modelo."""
        # aseguramos que el modelo esté en el dispositivo adecuado
        self.to(device)
        optimizer, scheduler = self.configure_optimizer(lr=lr)
        history = {"train_loss": [], "val_loss": []}
        best_loss = float("inf")

        print(f"Iniciando entrenamiento ({epochs} épocas)...")
        start = time.time()
        for ep in range(1, epochs + 1):
            # entrenamos y validamos en cada época
            t_loss = self.train_epoch(train_loader, optimizer, device)
            v_loss = self.validate_epoch(val_loader, device)

            # registramos la pérdida y ajustamos el planificador
            history["train_loss"].append(t_loss)
            history["val_loss"].append(v_loss)
            scheduler.step(v_loss)

            # guardamos los pesos si logramos el mejor resultado en validación
            if v_loss < best_loss and save_path:
                best_loss = v_loss
                self.save_weights(save_path)
                mark = " (mejor guardado)"
            else:
                mark = ""

            print(f"Época [{ep:02d}/{epochs:02d}] - Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}{mark}")

        mins = (time.time() - start) / 60
        print(f"Listo en {mins:.2f} minutos. Mejor Val Loss: {best_loss:.4f}")
        return history

    def save_weights(self, path: str):
        """Guarda los pesos del modelo en disco."""
        # creamos la carpeta contenedora si no existe
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str, device=None):
        """Carga los pesos guardados si el archivo existe."""
        # cargamos el diccionario de pesos desde el archivo
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, map_location=device))


class AgeEstimationModel(BaseFaceModel):
    """Modelo especialista para estimación de edad mediante regresión continua."""

    def __init__(self):
        super(AgeEstimationModel, self).__init__()
        # extractor convolucional para edad
        self.backbone = FeatureExtractorBackbone()
        # cabeza densa con dropout para predecir la edad normalizada
        self.age_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, 1),
        )
        # pérdida smooth l1 robusta ante edades avanzadas
        self.criterion = nn.SmoothL1Loss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Paso hacia adelante que extrae características y devuelve la edad normalizada."""
        features = self.backbone(x)
        # eliminamos la dimensión unitaria final para obtener un vector [B]
        return self.age_head(features).squeeze(1)

    def compute_loss(self, outputs, targets, device):
        """Calcula la pérdida de regresión de edad comparando con la etiqueta normalizada."""
        # comparamos la salida con la edad normalizada entre 0 y 1
        y = targets["age"].to(device)
        return self.criterion(outputs, y)

    def predict(self, x: torch.Tensor, device=None):
        """Genera la predicción de edad en años reales desnormalizada [1, 116]."""
        self.eval()
        if device:
            x = x.to(device)
            self.to(device)
        with torch.no_grad():
            out = self(x)
            # desnormalizamos multiplicando por 100 para regresar a años reales
            ages = out.cpu().numpy() * 100.0
            return np.clip(ages, 1.0, 116.0)


class GenderClassificationModel(BaseFaceModel):
    """Modelo especialista para clasificación binaria de género (0: Hombre, 1: Mujer)."""

    def __init__(self):
        super(GenderClassificationModel, self).__init__()
        # extractor convolucional para género
        self.backbone = FeatureExtractorBackbone()
        # cabeza densa que produce el logit antes de la sigmoide
        self.gender_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, 1),
        )
        # función bce con logits numéricamente estable
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Paso hacia adelante que produce el logit de clasificación de género."""
        features = self.backbone(x)
        # extraemos el logit escalar por imagen
        return self.gender_head(features).squeeze(1)

    def compute_loss(self, outputs, targets, device):
        """Calcula la pérdida de clasificación de género."""
        # comparamos el logit con la etiqueta binaria de género
        y = targets["gender"].to(device)
        return self.criterion(outputs, y)

    def predict(self, x: torch.Tensor, device=None):
        """Genera la probabilidad de ser mujer y la clase binaria asignada."""
        self.eval()
        if device:
            x = x.to(device)
            self.to(device)
        with torch.no_grad():
            logits = self(x)
            # aplicamos sigmoide para obtener la probabilidad de ser mujer
            probs = torch.sigmoid(logits).cpu().numpy()
            classes = (probs >= 0.5).astype(int)
            return probs, classes


class MultiTaskFaceModel(BaseFaceModel):
    """Modelo de aprendizaje multitarea con backbone compartido y dos cabezas de salida."""

    def __init__(self, weight_age: float = 1.0, weight_gender: float = 1.0):
        super(MultiTaskFaceModel, self).__init__()
        # extractor convolucional compartido por ambas tareas
        self.backbone = FeatureExtractorBackbone()

        # rama de regresión de edad
        self.age_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, 1),
        )

        # rama de clasificación de género
        self.gender_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, 1),
        )
        # función de costo combinada para ponderar las pérdidas
        self.criterion = MultiTaskLoss(weight_age=weight_age, weight_gender=weight_gender)

    def forward(self, x: torch.Tensor):
        """Paso hacia adelante que bifurca las características en dos ramas de salida."""
        # obtenemos el vector latente compartido y calculamos ambas salidas
        features = self.backbone(x)
        pred_age = self.age_head(features).squeeze(1)
        pred_gender = self.gender_head(features).squeeze(1)
        return pred_age, pred_gender

    def compute_loss(self, outputs, targets, device):
        """Calcula la pérdida combinada ponderando ambas tareas."""
        pred_age, pred_gender = outputs
        target_age = targets["age"].to(device)
        target_gender = targets["gender"].to(device)
        # calculamos el costo conjunto
        loss_total, _, _ = self.criterion(pred_age, pred_gender, target_age, target_gender)
        return loss_total

    def predict(self, x: torch.Tensor, device=None):
        """Inferencia conjunta devolviendo edad en años, probabilidad y clase de género."""
        self.eval()
        if device:
            x = x.to(device)
            self.to(device)
        with torch.no_grad():
            out_age, out_gen_logit = self(x)
            # desnormalizamos la edad y calculamos la probabilidad y clase de género
            ages = np.clip(out_age.cpu().numpy() * 100.0, 1.0, 116.0)
            probs = torch.sigmoid(out_gen_logit).cpu().numpy()
            classes = (probs >= 0.5).astype(int)
            return ages, probs, classes
