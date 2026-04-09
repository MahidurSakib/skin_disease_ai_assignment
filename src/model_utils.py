from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from src.utils import clean_class_name, save_json


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def save_class_names(raw_class_names: list[str], output_path: str | Path) -> list[str]:
    clean_names = [clean_class_name(name) for name in raw_class_names]
    save_json(clean_names, output_path)
    return clean_names


def load_checkpoint(model: nn.Module, checkpoint_path: str | Path, device: str | torch.device) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
