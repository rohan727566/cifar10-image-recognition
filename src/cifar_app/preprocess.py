"""Data preprocessing module for CIFAR-10 dataset."""

import logging
from typing import Tuple

import numpy as np
from keras.datasets import cifar10
from keras.utils import to_categorical
from PIL import Image
import io

from cifar_app.config import CLASS_NAMES, IMAGE_SIZE, NUM_CLASSES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_cifar() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load CIFAR-10 dataset from Keras datasets.

    Returns:
        Tuple containing:
            - (x_train, y_train): Training data and labels
            - (x_test, y_test): Test data and labels
    """
    logger.info("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    logger.info(f"Training samples: {x_train.shape[0]}")
    logger.info(f"Test samples: {x_test.shape[0]}")
    logger.info(f"Image shape: {x_train.shape[1:]}")
    
    return (x_train, y_train), (x_test, y_test)


def prepare_datasets(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare and normalize datasets for training.

    Args:
        x_train: Training images (raw pixel values 0-255)
        y_train: Training labels
        x_test: Test images (raw pixel values 0-255)
        y_test: Test labels

    Returns:
        Tuple of normalized and one-hot encoded data:
            (x_train_norm, y_train_cat, x_test_norm, y_test_cat)
    """
    logger.info("Normalizing pixel values...")
    # Normalize pixel values to [0, 1]
    x_train_norm = x_train.astype('float32') / 255.0
    x_test_norm = x_test.astype('float32') / 255.0
    
    logger.info("One-hot encoding labels...")
    # One-hot encode labels
    y_train_cat = to_categorical(y_train, NUM_CLASSES)
    y_test_cat = to_categorical(y_test, NUM_CLASSES)
    
    logger.info("Dataset preparation complete.")
    logger.info(f"Training data shape: {x_train_norm.shape}")
    logger.info(f"Training labels shape: {y_train_cat.shape}")
    
    return x_train_norm, y_train_cat, x_test_norm, y_test_cat


def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess uploaded image bytes for model inference.

    Args:
        image_bytes: Raw image bytes from uploaded file

    Returns:
        Preprocessed image array of shape (1, 32, 32, 3) ready for prediction
    """
    # Load image from bytes
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to 32x32
    image = image.resize((32, 32), Image.Resampling.LANCZOS)
    
    # Convert to numpy array and normalize
    image_array = np.array(image).astype('float32') / 255.0
    
    # Add batch dimension: (32, 32, 3) -> (1, 32, 32, 3)
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array


def get_class_names() -> list:
    """
    Get list of CIFAR-10 class names.

    Returns:
        List of 10 class names
    """
    return CLASS_NAMES


def get_label_from_class(class_index: int) -> str:
    """
    Convert class index to human-readable label.

    Args:
        class_index: Integer class index (0-9)

    Returns:
        String label (e.g., "airplane", "automobile")
    """
    if 0 <= class_index < NUM_CLASSES:
        return CLASS_NAMES[class_index]
    else:
        raise ValueError(f"Invalid class index: {class_index}. Must be 0-{NUM_CLASSES-1}")
