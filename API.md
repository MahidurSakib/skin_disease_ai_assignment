# API Documentation

## Endpoint

### POST `/analyze_skin`

Uploads a skin image, classifies the predicted disease, and returns LLM-based recommendations.

## Request

**Content-Type:** `multipart/form-data`

### Form fields
- `file`: image file (`jpg`, `jpeg`, `png`, `webp`, `bmp`)

## Response

**Status:** `200 OK`

```json
{
  "disease": "Eczema",
  "confidence": 0.9213,
  "recommendations": "Keep the area clean and avoid scratching. This is not a confirmed diagnosis.",
  "next_steps": "Monitor the area and consult a dermatologist if symptoms worsen or persist.",
  "tips": "Use good lighting for future photos and avoid irritants."
}
```

## Error responses

### `400 Bad Request`
- invalid image file type
- empty upload

### `503 Service Unavailable`
- trained model files are missing

### `500 Internal Server Error`
- unexpected prediction failure

## Interactive docs

When the FastAPI server is running:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
