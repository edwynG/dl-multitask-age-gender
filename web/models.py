import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Bloque convolucional básico compuesto por dos capas Conv2D con BatchNorm y ReLU, seguido de MaxPool2d."""

    def __init__(self, in_channels: int, out_channels: int):
        super(ConvBlock, self).__init__()
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
        return self.block(x)


class FeatureExtractorBackbone(nn.Module):
    """Extractor convolucional de características compartido (200x200x3 -> vector de 256 dimensiones)."""

    def __init__(self):
        super(FeatureExtractorBackbone, self).__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),    # 200x200 -> 100x100
            ConvBlock(32, 64),   # 100x100 -> 50x50
            ConvBlock(64, 128),  # 50x50   -> 25x25
            ConvBlock(128, 256), # 25x25   -> 12x12
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return x


class AgeEstimationModel(nn.Module):
    """Modelo especialista para estimación de edad (regresión continua)."""

    def __init__(self):
        super(AgeEstimationModel, self).__init__()
        self.backbone = FeatureExtractorBackbone()
        self.age_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        age = self.age_head(features)
        return age.squeeze(1)


class GenderClassificationModel(nn.Module):
    """Modelo especialista para clasificación de género (clasificación binaria)."""

    def __init__(self):
        super(GenderClassificationModel, self).__init__()
        self.backbone = FeatureExtractorBackbone()
        self.gender_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        gender_logit = self.gender_head(features)
        return gender_logit.squeeze(1)


class MultiTaskFaceModel(nn.Module):
    """Modelo de aprendizaje multitarea con backbone compartido y dos cabezas de predicción independientes."""

    def __init__(self):
        super(MultiTaskFaceModel, self).__init__()
        self.backbone = FeatureExtractorBackbone()

        self.age_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, 1),
        )

        self.gender_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        pred_age = self.age_head(features).squeeze(1)
        pred_gender = self.gender_head(features).squeeze(1)
        return pred_age, pred_gender
