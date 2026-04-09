from __future__ import annotations

from pydantic import BaseModel, Field


class AnalyzeSkinResponse(BaseModel):
    disease: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    recommendations: str
    next_steps: str
    tips: str
