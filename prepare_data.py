from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from src.utils import IMAGE_EXTENSIONS, ensure_dir, is_image_file, save_json


def resolve_raw_root(raw_dir: str | Path) -> Path:
    raw_path = Path(raw_dir)
    if (raw_path / "IMG_CLASSES").exists():
        return raw_path / "IMG_CLASSES"
    return raw_path


def list_class_directories(raw_root: Path) -> list[Path]:
    return sorted(
        [path for path in raw_root.iterdir() if path.is_dir() and any(is_image_file(p) for p in path.iterdir())]
    )


def split_items(items: list[Path], train_ratio: float, val_ratio: float, seed: int) -> tuple[list[Path], list[Path], list[Path]]:
    random.Random(seed).shuffle(items)
    total = len(items)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    train_items = items[:train_end]
    val_items = items[train_end:val_end]
    test_items = items[val_end:]
    return train_items, val_items, test_items


def copy_files(files: list[Path], destination_dir: Path) -> None:
    ensure_dir(destination_dir)
    for file_path in files:
        shutil.copy2(file_path, destination_dir / file_path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare skin disease dataset into train/val/test folders.")
    parser.add_argument("--raw-dir", required=True, help="Path to extracted dataset root or IMG_CLASSES folder.")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory for split dataset.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio <= 0:
        raise ValueError("train_ratio + val_ratio must be less than 1.0")

    raw_root = resolve_raw_root(args.raw_dir)
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw dataset path not found: {raw_root}")

    class_dirs = list_class_directories(raw_root)
    if not class_dirs:
        raise FileNotFoundError(
            f"No class folders with images were found inside: {raw_root}. "
            "Make sure you extracted the Kaggle zip file first."
        )

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    for split_name in ["train", "val", "test"]:
        split_dir = output_dir / split_name
        if split_dir.exists():
            shutil.rmtree(split_dir)

    split_summary: dict[str, dict[str, int]] = {}

    for class_dir in class_dirs:
        images = sorted([path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS])
        if not images:
            continue

        train_items, val_items, test_items = split_items(images, args.train_ratio, args.val_ratio, args.seed)

        copy_files(train_items, output_dir / "train" / class_dir.name)
        copy_files(val_items, output_dir / "val" / class_dir.name)
        copy_files(test_items, output_dir / "test" / class_dir.name)

        split_summary[class_dir.name] = {
            "train": len(train_items),
            "val": len(val_items),
            "test": len(test_items),
        }

    save_json(split_summary, output_dir / "split_summary.json")

    print("Dataset preparation completed.")
    print(f"Raw root: {raw_root}")
    print(f"Processed data saved to: {output_dir}")


if __name__ == "__main__":
    main()
