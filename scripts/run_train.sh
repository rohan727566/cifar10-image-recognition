#!/bin/bash
# Training script for CIFAR-10 model

set -e

echo "Starting CIFAR-10 model training..."

python -m src.cifar_app.train \
    --epochs ${EPOCHS:-18} \
    --batch-size ${BATCH_SIZE:-32} \
    --save-dir ${MODEL_PATH:-model/saved_model}

echo "Training completed successfully!"
