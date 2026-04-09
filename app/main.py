from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.schemas import AnalyzeSkinResponse
from src.classifier import get_classifier
from src.llm_service import LLMService

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
}

UNKNOWN_LABEL = "Unknown / Unsupported image"

app = FastAPI(
    title="Skin Disease Detection & LLM Advisor System",
    version="1.0.0",
    description="FastAPI backend for skin image classification and LLM-based recommendations.",
)

llm_service = LLMService()


@app.post("/analyze_skin", response_model=AnalyzeSkinResponse)
async def analyze_skin(file: UploadFile = File(...)) -> AnalyzeSkinResponse:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Please upload a valid skin image file.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        prediction = get_classifier().predict(image_bytes)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    disease = str(prediction["disease"])
    confidence = float(prediction["confidence"])

    if disease == UNKNOWN_LABEL:
        return AnalyzeSkinResponse(
            disease=disease,
            confidence=confidence,
            recommendations=(
                "The uploaded image does not confidently match the supported skin disease classes."
            ),
            next_steps=(
                "Please upload a clear close-up image of the affected skin area with good lighting."
            ),
            tips=(
                "Avoid uploading unrelated images such as bikes, screenshots, objects, or distant photos."
            ),
        )

    llm_output = llm_service.generate_recommendations(
        disease=disease,
        confidence=confidence,
    )

    return AnalyzeSkinResponse(
        disease=disease,
        confidence=confidence,
        recommendations=llm_output["recommendations"],
        next_steps=llm_output["next_steps"],
        tips=llm_output["tips"],
    )