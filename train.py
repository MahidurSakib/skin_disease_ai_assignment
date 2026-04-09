from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.model_utils import build_model, save_class_names
from src.preprocessing import get_eval_transform, get_train_transform
from src.training_utils import accuracy_from_logits, compute_class_weights, get_device
from src.utils import ensure_dir, save_json


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_accuracy = 0.0
    total_batches = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, labels)
            if is_training:
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item())
        total_accuracy += accuracy_from_logits(logits, labels)
        total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)
    avg_accuracy = total_accuracy / max(total_batches, 1)
    return avg_loss, avg_accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a ResNet18 skin disease classifier.")
    parser.add_argument("--data-dir", default="data/processed", help="Directory containing train/val/test folders.")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Directory to save model and metadata.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--freeze-backbone", action="store_true", help="Train only the final classification layer.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            "Processed train/val folders not found. Run prepare_data.py first."
        )

    device = get_device()
    artifacts_dir = ensure_dir(args.artifacts_dir)

    train_dataset = ImageFolder(train_dir, transform=get_train_transform(args.image_size))
    val_dataset = ImageFolder(val_dir, transform=get_eval_transform(args.image_size))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    class_names = save_class_names(train_dataset.classes, artifacts_dir / "class_names.json")

    model = build_model(num_classes=len(train_dataset.classes), pretrained=True)
    if args.freeze_backbone:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("fc.")
    model.to(device)

    class_weights = compute_class_weights(train_dataset.targets).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
    )

    best_val_accuracy = 0.0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)

        epoch_result = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_accuracy, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_accuracy, 4),
        }
        history.append(epoch_result)
        print(epoch_result)

        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "image_size": args.image_size,
            }
            torch.save(checkpoint, artifacts_dir / "best_model.pth")

    save_json(history, artifacts_dir / "train_history.json")
    print(f"Training completed. Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Model saved to: {artifacts_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()
