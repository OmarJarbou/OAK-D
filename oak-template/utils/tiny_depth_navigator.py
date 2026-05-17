# utils/tiny_depth_navigator.py
"""Lightweight depth CNN for walker navigation (4 classes)."""

from __future__ import annotations

import torch
import torch.nn as nn

NAV_CLASSES = ("LEFT", "CENTER", "RIGHT", "STOP")
CLASS_TO_IDX = {name: i for i, name in enumerate(NAV_CLASSES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(NAV_CLASSES)}


class TinyDepthNavigator(nn.Module):
    """Depth-only CNN (~500–1000 params). Input: 1 x H x W (normalized depth)."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(32 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.drop = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)
