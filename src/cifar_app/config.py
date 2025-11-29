"""Configuration settings for CIFAR-10 image recognition system."""

import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "model"
DOCS_DIR = BASE_DIR / "docs"
PLOTS_DIR = DOCS_DIR / "plots"

# Model configuration
MODEL_PATH = os.getenv("MODEL_PATH", str(MODEL_DIR / "saved_model"))
WEIGHTS_PATH = MODEL_DIR / "weights.h5"
HISTORY_PATH = MODEL_DIR / "history.json"

# Training configuration
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
EPOCHS = int(os.getenv("EPOCHS", "18"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
VALIDATION_SPLIT = 0.2

# Dataset configuration
IMAGE_SIZE = (32, 32, 3)
NUM_CLASSES = 10
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Server configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
WORKERS = int(os.getenv("WORKERS", "1"))
RELOAD = os.getenv("RELOAD", "false").lower() == "true"

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Create directories if they don't exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
