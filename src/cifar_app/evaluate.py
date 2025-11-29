"""Model evaluation and metrics generation."""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model

from cifar_app.preprocess import load_cifar, prepare_datasets, get_class_names

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    """Plots and saves confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues",
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    plt.savefig(output_path / "confusion_matrix.png")
    plt.close()
    logger.info(f"Confusion matrix saved to {output_path / 'confusion_matrix.png'}")


def save_classification_report(y_true, y_pred, class_names, output_path):
    """Saves classification report as JSON."""
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )
    with open(output_path / "classification_report.json", "w") as f:
        json.dump(report_dict, f, indent=2)
    logger.info(f"Classification report saved to {output_path / 'classification_report.json'}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate CIFAR-10 model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to saved model directory or file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="docs/plots",
        help="Directory to save evaluation plots and reports",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model from {args.model}")
    model = load_model(args.model)

    logger.info("Loading and preparing CIFAR-10 data")
    (x_train, y_train), (x_test, y_test) = load_cifar()
    _, _, x_test_norm, y_test_cat = prepare_datasets(x_train, y_train, x_test, y_test)

    logger.info("Predicting test data...")
    y_pred_probs = model.predict(x_test_norm, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = y_test.flatten()

    class_names = get_class_names()

    save_classification_report(y_true, y_pred, class_names, output_path)
    plot_confusion_matrix(y_true, y_pred, class_names, output_path)

    logger.info("Evaluation completed successfully.")


if __name__ == "__main__":
    main()
