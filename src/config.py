from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    image_size: int = int(os.getenv("IMAGE_SIZE", "224"))
    model_path: str = os.getenv("MODEL_PATH", "artifacts/best_model.pth")
    class_names_path: str = os.getenv("CLASS_NAMES_PATH", "artifacts/class_names.json")
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
    api_base_url: str = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


settings = Settings()
