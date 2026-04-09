from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def is_image_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def clean_class_name(raw_name: str) -> str:
    """
    Converts noisy Kaggle folder names into cleaner labels.

    Examples:
    - '1. Eczema 1677' -> 'Eczema'
    - '10. Warts Molluscum and other Viral Infections - 2103' ->
      'Warts Molluscum and other Viral Infections'
    """
    name = raw_name.strip()
    name = re.sub(r"^\d+\.?\s*", "", name)
    name = re.sub(r"\s*-\s*\d+(?:\.\d+)?k?$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s+\d+(?:\.\d+)?k?$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s+", " ", name).strip(" -")
    return name


def save_json(data: Any, path: str | Path) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)
