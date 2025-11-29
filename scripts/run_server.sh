#!/bin/bash
# Server startup script

set -e

echo "Starting CIFAR-10 inference server..."

uvicorn src.cifar_app.server:app \
    --host ${HOST:-0.0.0.0} \
    --port ${PORT:-8000} \
    --reload
