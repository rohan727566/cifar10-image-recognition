"""Model definition and loading utilities for CIFAR-10 CNN."""

import logging
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from cifar_app.config import IMAGE_SIZE, NUM_CLASSES
from cifar_app.preprocess import get_label_from_class

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_model() -> Sequential:
    """
    Build CNN model architecture for CIFAR-10 classification.

    Architecture:
        - Conv2D(32, 3x3, relu) -> Conv2D(32, 3x3, relu) -> MaxPool2D -> Dropout(0.25)
        - Conv2D(64, 3x3, relu) -> Conv2D(64, 3x3, relu) -> MaxPool2D -> Dropout(0.25)
        - Flatten -> Dense(512, relu) -> Dropout(0.5) -> Dense(10, softmax)

    Returns:
        Compiled Keras Sequential model
    """
    logger.info("Building CNN model architecture...")
    
    model = Sequential()
    
    # First convolutional block
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', 
                     input_shape=IMAGE_SIZE))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Second convolutional block
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    
    # Compile model
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    logger.info("Model architecture built successfully.")
    logger.info(f"Total parameters: {model.count_params():,}")
    
    return model


def load_saved_model(model_path: str) -> Sequential:
    """
    Load a saved Keras model from disk.

    Args:
        model_path: Path to saved model directory or .h5 file

    Returns:
        Loaded Keras model

    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    model_path_obj = Path(model_path)
    
    if not model_path_obj.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    logger.info(f"Loading model from: {model_path}")
    model = load_model(model_path)
    logger.info("Model loaded successfully.")
    
    return model


def predict_from_array(
    model: Sequential,
    image_array: np.ndarray,
    top_k: int = 3
) -> Dict:
    """
    Make prediction from preprocessed image array.

    Args:
        model: Trained Keras model
        image_array: Preprocessed image array of shape (1, 32, 32, 3)
        top_k: Number of top predictions to return

    Returns:
        Dictionary containing:
            - predicted_class: Top predicted class index
            - predicted_label: Top predicted class name
            - confidence: Confidence score (0-1)
            - top_k_predictions: List of top-k predictions with class, label, and probability
    """
    # Get predictions
    predictions = model.predict(image_array, verbose=0)
    probabilities = predictions[0]  # Shape: (10,)
    
    # Get top prediction
    predicted_class = int(np.argmax(probabilities))
    predicted_label = get_label_from_class(predicted_class)
    confidence = float(probabilities[predicted_class])
    
    # Get top-k predictions
    top_k_indices = np.argsort(probabilities)[::-1][:top_k]
    top_k_predictions = []
    
    for idx in top_k_indices:
        top_k_predictions.append({
            "class": int(idx),
            "label": get_label_from_class(int(idx)),
            "probability": float(probabilities[idx])
        })
    
    return {
        "predicted_class": predicted_class,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "top_predictions": top_k_predictions
    }


def predict_from_bytes(
    model: Sequential,
    image_bytes: bytes,
    top_k: int = 3
) -> Dict:
    """
    Make prediction from raw image bytes.

    Args:
        model: Trained Keras model
        image_bytes: Raw image bytes
        top_k: Number of top predictions to return

    Returns:
        Dictionary with prediction results (same format as predict_from_array)
    """
    from cifar_app.preprocess import preprocess_image_bytes
    
    # Preprocess image
    image_array = preprocess_image_bytes(image_bytes)
    
    # Get predictions
    return predict_from_array(model, image_array, top_k)
