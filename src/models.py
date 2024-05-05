import torch
import torch.nn as nn

from torchvision.models import get_model

class BasicClassifier(nn.Module):
    def __init__(self, n_classes, model_name, pretrained=True):
        super(BasicClassifier, self).__init__()
        weights = 'DEFAULT' if pretrained else None
        self.backbone = get_model(model_name, weights=weights).features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2, inplace=True),
            nn.Linear(1280, n_classes)
            )
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x).squeeze(dim=(-1,-2))
        x = self.classifier(x)
        return x