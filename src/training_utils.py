from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import accuracy_score


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_class_weights(targets: Iterable[int]) -> torch.Tensor:
    targets = list(targets)
    class_counts = np.bincount(targets)
    class_counts[class_counts == 0] = 1
    weights = len(targets) / (len(class_counts) * class_counts)
    return torch.tensor(weights, dtype=torch.float32)


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = torch.argmax(logits, dim=1).cpu().numpy()
    labels_np = labels.cpu().numpy()
    return float(accuracy_score(labels_np, predictions))
