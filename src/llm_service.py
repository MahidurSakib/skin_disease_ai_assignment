from __future__ import annotations

import json
import re
from typing import Any

from src.config import settings

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


class LLMService:
    def __init__(self) -> None:
        self.provider = settings.llm_provider.lower().strip()
        self.model = settings.openai_model
        self.api_key = settings.openai_api_key
        self.client = None

        if self.provider == "openai" and self.api_key and OpenAI is not None:
            self.client = OpenAI(api_key=self.api_key)

    def generate_recommendations(self, disease: str, confidence: float) -> dict[str, str]:
        if self.client is None:
            return self._fallback_response(disease, confidence)

        prompt = f"""
Predicted disease: {disease}
Confidence score: {confidence:.4f}

Return only valid JSON with exactly these keys:
- recommendations
- next_steps
- tips

Rules:
1. Keep the answer short and practical.
2. Do not claim a confirmed diagnosis.
3. Do not prescribe medication dosage.
4. Mention when to seek a dermatologist.
5. Use plain English.
6. Make the content specific to the predicted disease.
7. If the disease is melanoma, make the next steps more urgent.
8. If the disease is eczema or atopic dermatitis, mention moisturising and trigger avoidance.
9. If the disease is acne, mention gentle cleansing and avoiding picking.
10. If the disease is psoriasis, mention flare monitoring and skin hydration.
11. Each section must be different depending on the disease label.
""".strip()

        try:
            response = self.client.responses.create(
                model=self.model,
                reasoning={"effort": "none"},
                text={"verbosity": "low"},
                instructions=(
                    "You are a careful dermatology support assistant for an academic demo. "
                    "Based on a classifier label and confidence, provide general educational advice only. "
                    "The advice must vary depending on the disease label. "
                    "Never present the output as a confirmed diagnosis. Output strict JSON only."
                ),
                input=prompt,
            )
            output_text = getattr(response, "output_text", "") or ""
            return self._parse_json_response(output_text, disease, confidence)
        except Exception:
            return self._fallback_response(disease, confidence)

    def _parse_json_response(
        self,
        response_text: str,
        disease: str,
        confidence: float,
    ) -> dict[str, str]:
        try:
            match = re.search(r"\{.*\}", response_text, flags=re.DOTALL)
            json_text = match.group(0) if match else response_text
            data: dict[str, Any] = json.loads(json_text)

            recommendations = str(data.get("recommendations", "")).strip()
            next_steps = str(data.get("next_steps", "")).strip()
            tips = str(data.get("tips", "")).strip()

            if recommendations and next_steps and tips:
                return {
                    "recommendations": recommendations,
                    "next_steps": next_steps,
                    "tips": tips,
                }
        except Exception:
            pass

        return self._fallback_response(disease, confidence)

    def _fallback_response(self, disease: str, confidence: float) -> dict[str, str]:
        percent = round(confidence * 100, 2)
        disease_key = disease.lower().strip()

        disease_guidance = {
            "melanoma": {
                "recommendations": (
                    f"The image model suggests {disease} with {percent}% confidence. "
                    "This may be a serious condition, so avoid relying on self-treatment and seek professional evaluation."
                ),
                "next_steps": (
                    "Arrange a dermatologist appointment as soon as possible, especially if the lesion is changing in size, shape, color, or is bleeding."
                ),
                "tips": (
                    "Take clear photos over time, avoid scratching or irritating the area, and monitor for rapid visible changes."
                ),
            },
            "eczema": {
                "recommendations": (
                    f"The image model suggests {disease} with {percent}% confidence. "
                    "Focus on moisturising the skin and avoiding harsh soaps, fragrances, and known irritants."
                ),
                "next_steps": (
                    "Monitor itching, redness, dryness, or spread. Seek medical advice if the area becomes painful, infected, or does not improve."
                ),
                "tips": (
                    "Use gentle skincare products, avoid scratching, and note any trigger such as soap, dust, heat, or fabric."
                ),
            },
            "psoriasis": {
                "recommendations": (
                    f"The image model suggests {disease} with {percent}% confidence. "
                    "Keep the skin moisturised and avoid known flare triggers such as stress, irritation, and skin injury."
                ),
                "next_steps": (
                    "Consult a dermatologist if plaques spread, crack, bleed, or become increasingly uncomfortable."
                ),
                "tips": (
                    "Track flare patterns, avoid aggressive scrubbing, and use good lighting when monitoring changes."
                ),
            },
            "acne": {
                "recommendations": (
                    f"The image model suggests {disease} with {percent}% confidence. "
                    "Use gentle cleansing and avoid squeezing, picking, or over-irritating the affected area."
                ),
                "next_steps": (
                    "Seek medical advice if breakouts are severe, painful, widespread, or causing scarring."
                ),
                "tips": (
                    "Wash gently, avoid comedogenic products, and monitor whether food, stress, or skincare products worsen the condition."
                ),
            },
            "atopic dermatitis": {
                "recommendations": (
                    f"The image model suggests {disease} with {percent}% confidence. "
                    "Support the skin barrier with frequent moisturising and avoid triggers that may increase irritation."
                ),
                "next_steps": (
                    "Monitor itching, rash spread, and skin cracking. See a dermatologist if symptoms worsen or infection appears."
                ),
                "tips": (
                    "Use fragrance-free products, keep nails short to reduce scratching damage, and track possible triggers."
                ),
            },
            "basal cell carcinoma": {
                "recommendations": (
                    f"The image model suggests {disease} with {percent}% confidence. "
                    "This type of lesion should be assessed by a dermatologist rather than managed with home treatment alone."
                ),
                "next_steps": (
                    "Arrange a dermatology visit, especially if the lesion persists, grows, crusts, or bleeds."
                ),
                "tips": (
                    "Take clear progress photos, avoid irritating the area, and note any visible change in texture or size."
                ),
            },
            "benign keratosis-like lesions": {
                "recommendations": (
                    f"The image model suggests {disease} with {percent}% confidence. "
                    "These lesions are often monitored, but any rapid change should still be evaluated professionally."
                ),
                "next_steps": (
                    "Watch for changes in size, color, thickness, or irritation, and seek medical review if changes occur."
                ),
                "tips": (
                    "Track the lesion over time with photos and avoid scratching or repeated friction on the area."
                ),
            },
            "dermatofibroma": {
                "recommendations": (
                    f"The image model suggests {disease} with {percent}% confidence. "
                    "This type of bump is often benign, but it should still be reviewed if it becomes painful or changes noticeably."
                ),
                "next_steps": (
                    "Monitor for growth, color change, pain, or irritation, and consult a dermatologist if any of those appear."
                ),
                "tips": (
                    "Avoid picking the lesion and keep simple photo records if you notice gradual changes."
                ),
            },
            "fungal infections": {
                "recommendations": (
                    f"The image model suggests {disease} with {percent}% confidence. "
                    "Keep the area clean and dry, and avoid sharing towels, clothing, or personal skin-contact items."
                ),
                "next_steps": (
                    "Seek medical advice if the rash spreads, becomes painful, or does not improve over time."
                ),
                "tips": (
                    "Maintain hygiene, reduce moisture buildup, and monitor whether the affected area enlarges or changes shape."
                ),
            },
            "warts molluscum and other viral infections": {
                "recommendations": (
                    f"The image model suggests {disease} with {percent}% confidence. "
                    "Avoid scratching or spreading the area, and keep the affected skin clean."
                ),
                "next_steps": (
                    "Consult a dermatologist if lesions multiply, become painful, or persist for a long period."
                ),
                "tips": (
                    "Do not pick the lesions, avoid direct contact with shared items, and track any increase in number or size."
                ),
            },
        }

        for key, value in disease_guidance.items():
            if key in disease_key:
                return value

        return {
            "recommendations": (
                f"The image model suggests {disease} with {percent}% confidence. "
                "Keep the area clean and avoid self-diagnosing based only on the image result."
            ),
            "next_steps": (
                "Monitor the area and consult a dermatologist if symptoms worsen, spread, bleed, become painful, or remain persistent."
            ),
            "tips": (
                "Take clear photos in good lighting, avoid harsh products, and keep track of visible changes over time."
            ),
        }