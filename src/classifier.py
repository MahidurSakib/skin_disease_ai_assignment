from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from src.config import settings
from src.model_utils import build_model, load_checkpoint
from src.preprocessing import get_eval_transform, load_rgb_image
from src.utils import load_json


class SkinDiseaseClassifier:
    def __init__(self, model_path: str, class_names_path: str, image_size: int = 224) -> None:
        self.model_path = Path(model_path)
        self.class_names_path = Path(class_names_path)
        self.image_size = image_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at '{self.model_path}'. Train the model first using train.py."
            )
        if not self.class_names_path.exists():
            raise FileNotFoundError(
                f"Class names file not found at '{self.class_names_path}'. Train the model first using train.py."
            )

        self.class_names = load_json(self.class_names_path)
        self.transform = get_eval_transform(self.image_size)
        self.model = build_model(num_classes=len(self.class_names), pretrained=False)
        self.model = load_checkpoint(self.model, self.model_path, self.device)

    def _has_sufficient_skin_region(self, image) -> bool:
        """
        Simple RGB-based skin detection heuristic.
        This helps reject clearly unrelated images like bikes, cats, screenshots, rooms, etc.
        """
        img = np.array(image.convert("RGB"))
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]

        skin_mask = (
            (r > 95) &
            (g > 40) &
            (b > 20) &
            ((np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)) > 15) &
            (np.abs(r - g) > 15) &
            (r > g) &
            (r > b)
        )

        skin_ratio = skin_mask.mean()

        # You can tune this if needed.
        # 0.08 means at least ~8% of the image should look like skin.
        return skin_ratio >= 0.08

    def predict(self, image_bytes: bytes) -> dict[str, float | str]:
        image = load_rgb_image(image_bytes)

        # Step 1: Reject obviously unrelated non-skin images
        if not self._has_sufficient_skin_region(image):
            return {
                "disease": "Unknown / Unsupported image",
                "confidence": 0.0,
            }

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        top_probs, top_indices = torch.topk(probabilities, k=2)

        top1_conf = float(top_probs[0].item())
        top2_conf = float(top_probs[1].item())
        predicted_idx = int(top_indices[0].item())
        disease = self.class_names[predicted_idx]

        # Step 2: Confidence-based rejection as extra protection
        confidence_threshold = 0.60
        margin_threshold = 0.15

        is_unknown = (
            top1_conf < confidence_threshold
            or (top1_conf - top2_conf) < margin_threshold
        )

        if is_unknown:
            return {
                "disease": "Unknown / Unsupported image",
                "confidence": round(top1_conf, 4),
            }

        return {
            "disease": disease,
            "confidence": round(top1_conf, 4),
        }


@lru_cache(maxsize=1)
def get_classifier() -> SkinDiseaseClassifier:
    return SkinDiseaseClassifier(
        model_path=settings.model_path,
        class_names_path=settings.class_names_path,
        image_size=settings.image_size,
    )