"""
CNN architectures for Premier League logo classification.

Models:
    - SimpleCNN: 3 conv blocks (baseline)
    - DeepCNN: 5 conv blocks with skip connections and GAP
    - get_resnet50: Pretrained ResNet50 with configurable freezing
"""

import torch
import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    """
    Shallow CNN with 3 convolutional blocks.

    Architecture:
        3 x [Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d]
        Filter progression: 32 -> 64 -> 128
        Classifier: Flatten -> Linear(512) -> ReLU -> Dropout -> Linear(num_classes)

    Input: 3 x 128 x 128
    """

    def __init__(self, num_classes=20):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 128x128 -> 64x64
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 64x64 -> 32x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        # After 3 pools: 128 / 8 = 16 -> feature map is 128 x 16 x 16
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class DeepCNN(nn.Module):
    """
    Deeper CNN with 5 conv blocks, skip connections, and Global Average Pooling.

    Architecture:
        5 x [Conv2d -> (BatchNorm2d) -> ReLU -> MaxPool2d]
        Filter progression: 32 -> 64 -> 128 -> 256 -> 256
        Skip connections (addition) at blocks 3 and 5
        Classifier: GAP -> Linear(256) -> ReLU -> Dropout -> Linear(num_classes)

    Args:
        num_classes: Number of output classes
        use_batchnorm: Whether to include BatchNorm layers (for ablation)
        dropout_rate: Dropout rate in the classifier head (for ablation)

    Input: 3 x 128 x 128
    """

    def __init__(self, num_classes=20, use_batchnorm=True, dropout_rate=0.5):
        super().__init__()
        self.use_batchnorm = use_batchnorm

        # Block 1: 3 -> 32
        self.block1 = self._make_block(3, 32)
        # Block 2: 32 -> 64
        self.block2 = self._make_block(32, 64)
        # Block 3: 64 -> 128 (with skip connection)
        self.block3 = self._make_block(64, 128)
        self.skip3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.MaxPool2d(2, 2),
        )
        # Block 4: 128 -> 256
        self.block4 = self._make_block(128, 256)
        # Block 5: 256 -> 256 (with skip connection)
        self.block5 = self._make_block(256, 256)
        self.skip5 = nn.MaxPool2d(2, 2)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def _make_block(self, in_ch, out_ch):
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)]
        if self.use_batchnorm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(2, 2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.block1(x)       # -> 32 x 64 x 64
        x = self.block2(x)       # -> 64 x 32 x 32

        identity3 = self.skip3(x)
        x = self.block3(x)       # -> 128 x 16 x 16
        x = x + identity3        # skip connection

        x = self.block4(x)       # -> 256 x 8 x 8

        identity5 = self.skip5(x)
        x = self.block5(x)       # -> 256 x 4 x 4
        x = x + identity5        # skip connection

        x = self.gap(x)          # -> 256 x 1 x 1
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def get_resnet50(num_classes, freeze_mode='full_freeze'):
    """
    Load pretrained ResNet50 and apply a freezing strategy.

    Args:
        num_classes: Number of output classes
        freeze_mode: One of:
            'full_freeze'    — freeze all layers, train only FC head
            'partial'        — freeze up to layer3, unfreeze layer4 + FC
            'full_finetune'  — unfreeze everything

    Returns:
        Modified ResNet50 model
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    if freeze_mode == 'full_freeze':
        for param in model.parameters():
            param.requires_grad = False

    elif freeze_mode == 'partial':
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = True

    elif freeze_mode == 'full_finetune':
        pass  # all params remain trainable

    # Replace classification head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )

    return model
