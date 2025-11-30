"""Unit tests for model and preprocessing functions."""

import numpy as np
import pytest
from cifar_app.preprocess import (
    get_class_names,
    get_label_from_class,
    preprocess_image_bytes,
)
from cifar_app.model import build_model


def test_get_class_names():
    """Test that class names are returned correctly."""
    class_names = get_class_names()
    assert len(class_names) == 10
    assert "airplane" in class_names
    assert "cat" in class_names


def test_get_label_from_class():
    """Test conversion from class index to label."""
    assert get_label_from_class(0) == "airplane"
    assert get_label_from_class(3) == "cat"
    assert get_label_from_class(9) == "truck"

    with pytest.raises(ValueError):
        get_label_from_class(10)  # Invalid class


def test_preprocess_image_bytes():
    """Test image preprocessing from bytes."""
    # Create a synthetic 32x32 RGB image
    from PIL import Image
    import io

    img = Image.new("RGB", (32, 32), color="red")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()

    # Preprocess
    processed = preprocess_image_bytes(img_bytes)

    # Check shape
    assert processed.shape == (1, 32, 32, 3)
    # Check normalization (values should be between 0 and 1)
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0


def test_build_model():
    """Test that model architecture builds correctly."""
    model = build_model()

    # Check model has layers
    assert len(model.layers) > 0

    # Check input shape
    assert model.input_shape == (None, 32, 32, 3)

    # Check output shape (10 classes)
    assert model.output_shape == (None, 10)

    # Check model is compiled
    assert model.optimizer is not None


def test_model_prediction_shape():
    """Test that model prediction returns correct shape."""
    model = build_model()

    # Create synthetic input
    test_input = np.random.rand(1, 32, 32, 3).astype("float32")

    # Get prediction
    prediction = model.predict(test_input, verbose=0)

    # Check output shape
    assert prediction.shape == (1, 10)

    # Check probabilities sum to ~1
    assert np.isclose(prediction.sum(), 1.0, atol=0.01)
