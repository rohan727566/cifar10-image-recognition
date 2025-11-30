"""API endpoint tests using FastAPI TestClient."""

import io
from PIL import Image
from fastapi.testclient import TestClient
from src.cifar_app.server import app

client = TestClient(app)


def test_health_endpoint():
    """Test /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_with_valid_image():
    """Test /predict endpoint with valid image."""
    # Create synthetic 32x32 image
    img = Image.new("RGB", (32, 32), color="blue")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    # Send request
    response = client.post(
        "/predict",
        files={"file": ("test.png", img_bytes, "image/png")},
    )

    assert response.status_code == 200
    data = response.json()

    # Check response structure
    assert "predicted_class" in data
    assert "predicted_label" in data
    assert "confidence" in data
    assert "top3" in data
    assert "inference_time_ms" in data

    # Check data types
    assert isinstance(data["predicted_class"], int)
    assert isinstance(data["predicted_label"], str)
    assert isinstance(data["confidence"], float)
    assert len(data["top3"]) == 3

    # Check value ranges
    assert 0 <= data["predicted_class"] <= 9
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_endpoint_without_file():
    """Test /predict endpoint without file."""
    response = client.post("/predict")
    assert response.status_code == 422  # Unprocessable Entity


def test_predict_endpoint_with_invalid_file():
    """Test /predict endpoint with non-image file."""
    # Send text file instead of image
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    # Should return 400 or similar error
    assert response.status_code in [400, 422, 500]
