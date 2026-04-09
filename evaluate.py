from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.model_utils import build_model, load_checkpoint
from src.preprocessing import get_eval_transform
from src.training_utils import get_device
from src.utils import load_json, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained model on the test split.")
    parser.add_argument("--data-dir", default="data/processed", help="Directory containing train/val/test folders.")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Directory containing model files.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    test_dir = data_dir / "test"
    artifacts_dir = Path(args.artifacts_dir)
    model_path = artifacts_dir / "best_model.pth"
    class_names_path = artifacts_dir / "class_names.json"

    if not test_dir.exists():
        raise FileNotFoundError("Test folder not found. Run prepare_data.py first.")
    if not model_path.exists():
        raise FileNotFoundError("Model checkpoint not found. Run train.py first.")
    if not class_names_path.exists():
        raise FileNotFoundError("Class names file not found. Run train.py first.")

    device = get_device()
    class_names = load_json(class_names_path)

    test_dataset = ImageFolder(test_dir, transform=get_eval_transform(args.image_size))
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(num_classes=len(test_dataset.classes), pretrained=False)
    model = load_checkpoint(model, model_path, device)

    all_labels: list[int] = []
    all_predictions: list[int] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            predictions = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            all_predictions.extend(predictions)
            all_labels.extend(labels.numpy().tolist())

    accuracy = float(accuracy_score(all_labels, all_predictions))
    report = classification_report(all_labels, all_predictions, target_names=class_names, output_dict=True)
    matrix = confusion_matrix(all_labels, all_predictions)

    metrics = {
        "accuracy": round(accuracy, 4),
        "classification_report": report,
    }
    save_json(metrics, artifacts_dir / "metrics.json")

    plt.figure(figsize=(12, 10))
    plt.imshow(matrix, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(artifacts_dir / "confusion_matrix.png", bbox_inches="tight")
    plt.close()

    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Metrics saved to: {artifacts_dir / 'metrics.json'}")
    print(f"Confusion matrix saved to: {artifacts_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
