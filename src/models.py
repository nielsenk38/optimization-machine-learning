"""Model definitions used in the experiments."""

from __future__ import annotations

import torch
from torch import nn


class SmallCNN(nn.Module):
    """A compact CNN for Fashion-MNIST.

    The model intentionally avoids BatchNorm and Dropout so that the main
    experimental variable remains the training-data order, not additional
    regularization or batch-statistics effects.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class LogisticRegression(nn.Module):
    """A convex baseline model for optional sanity checks."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.linear = nn.Linear(28 * 28, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.view(x.size(0), -1))


def build_model(model_name: str) -> nn.Module:
    """Create a model by name."""
    if model_name == "small_cnn":
        return SmallCNN()
    if model_name == "logistic_regression":
        return LogisticRegression()
    raise ValueError(f"Unknown model: {model_name}")
