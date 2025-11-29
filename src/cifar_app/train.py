"""Training script for CIFAR-10 CNN model."""

import logging
import json
import argparse
from pathlib import Path
from datetime import datetime

from keras.callbacks import ModelCheckpoint, EarlyStopping

from cifar_app.config import (
    BATCH_SIZE,
    EPOCHS,
    VALIDATION_SPLIT,
    MODEL_DIR,
    WEIGHTS_PATH,
    HISTORY_PATH
)
from cifar_app.preprocess import load_cifar, prepare_datasets
from cifar_app.model import build_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_model(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    save_dir: str = None,
    validation_split: float = VALIDATION_SPLIT
) -> dict:
    """
    Train CIFAR-10 CNN model.

    Args:
        epochs: Number of training epochs
        batch_size: Training batch size
        save_dir: Directory to save trained model
        validation_split: Fraction of training data to use for validation

    Returns:
        Dictionary containing training history
    """
    logger.info("=" * 70)
    logger.info("CIFAR-10 CNN Training Pipeline")
    logger.info("=" * 70)
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Validation split: {validation_split}")
    logger.info("=" * 70)
    
    # Load data
    (x_train, y_train), (x_test, y_test) = load_cifar()
    
    # Prepare datasets
    x_train_norm, y_train_cat, x_test_norm, y_test_cat = prepare_datasets(
        x_train, y_train, x_test, y_test
    )
    
    # Build model
    model = build_model()
    
    # Print model summary
    logger.info("\nModel Architecture:")
    model.summary(print_fn=logger.info)
    
    # Setup callbacks
    callbacks = []
    
    # Model checkpoint - save best model
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = save_path / "best_model.keras"
        checkpoint = ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        callbacks.append(checkpoint)
        logger.info(f"Model checkpoint: {checkpoint_path}")
    
    # Early stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stop)
    
    # Train model
    logger.info("\nStarting training...")
    logger.info("=" * 70)
    
    start_time = datetime.now()
    
    history = model.fit(
        x_train_norm,
        y_train_cat,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1
    )
    
    end_time = datetime.now()
    training_duration = end_time - start_time
    
    logger.info("=" * 70)
    logger.info(f"Training completed in: {training_duration}")
    
    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(x_test_norm, y_test_cat, verbose=0)
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Save final model and weights
    if save_dir:
        save_path = Path(save_dir)
        
        # Save in SavedModel format
        final_model_path = save_path / "saved_model"
        model.save(str(final_model_path))
        logger.info(f"Final model saved: {final_model_path}")
        
        # Save weights
        weights_path = save_path / "weights.h5"
        model.save_weights(str(weights_path))
        logger.info(f"Model weights saved: {weights_path}")
        
        # Save training history
        history_dict = {
            'loss': [float(x) for x in history.history['loss']],
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'epochs_trained': len(history.history['loss']),
            'training_duration_seconds': training_duration.total_seconds(),
            'config': {
                'epochs': epochs,
                'batch_size': batch_size,
                'validation_split': validation_split
            }
        }
        
        history_path = save_path / "history.json"
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        logger.info(f"Training history saved: {history_path}")
    
    logger.info("=" * 70)
    logger.info("Training pipeline completed successfully!")
    logger.info("=" * 70)
    
    return history.history


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(description='Train CIFAR-10 CNN model')
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=EPOCHS,
        help=f'Number of training epochs (default: {EPOCHS})'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=BATCH_SIZE,
        help=f'Training batch size (default: {BATCH_SIZE})'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default=str(MODEL_DIR),
        help=f'Directory to save trained model (default: {MODEL_DIR})'
    )
    parser.add_argument(
        '--validation-split',
        type=float,
        default=VALIDATION_SPLIT,
        help=f'Validation split fraction (default: {VALIDATION_SPLIT})'
    )
    
    args = parser.parse_args()
    
    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        validation_split=args.validation_split
    )


if __name__ == '__main__':
    main()
