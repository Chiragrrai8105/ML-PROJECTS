import torch
import torch.nn as nn
from torchvision import models


class AgeGenderModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = models.efficientnet_b0(
            weights=None
        )

        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Identity()

        self.age_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

        self.gender_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):

        features = self.backbone(x)

        age = self.age_head(features)

        gender = self.gender_head(features)

        return age, gender