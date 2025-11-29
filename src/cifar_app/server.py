"""FastAPI inference server for CIFAR-10 image recognition."""

import os
import time
import logging
from typing import List, Dict
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from pydantic import BaseModel

from cifar_app.model import load_saved_model, predict_from_bytes
from cifar_app.config import MODEL_PATH, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

app = FastAPI(title="CIFAR-10 Image Recognition API")

# Enable CORS for local web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for dev; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model globally on startup
MODEL_FILE_PATH = os.getenv("MODEL_PATH", MODEL_PATH)

try:
    model = load_saved_model(MODEL_FILE_PATH)
    logger.info(f"Model loaded from {MODEL_FILE_PATH}")
except Exception as e:
    logger.error(f"Failed to load model from {MODEL_FILE_PATH}: {e}")
    model = None

class PredictionResult(BaseModel):
    predicted_class: int
    predicted_label: str
    confidence: float
    top3: List[Dict[str, object]]
    preprocessing_info: str
    inference_time_ms: float

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResult)
async def predict(file: UploadFile = File(...), request: Request = None):
    """Make prediction on uploaded image."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        contents = await file.read()
        start_time = time.time()
        prediction = predict_from_bytes(model, contents, top_k=3)
        inference_time = (time.time() - start_time) * 1000  # ms

        result = PredictionResult(
            predicted_class=prediction["predicted_class"],
            predicted_label=prediction["predicted_label"],
            confidence=prediction["confidence"],
            top3=prediction["top_predictions"],
            preprocessing_info="Image resized to 32x32, normalized.",
            inference_time_ms=inference_time,
        )
        return result
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

def main():
    import uvicorn
    uvicorn.run(
        "src.cifar_app.server:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )

if __name__ == "__main__":
    main()
