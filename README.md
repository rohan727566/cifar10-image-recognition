# 🖼️ CIFAR-10 Image Recognition System

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://tensorflow.org/)
[![Keras 2.15](https://img.shields.io/badge/Keras-2.15-red.svg)](https://keras.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub Actions CI](https://github.com/rohan727566/cifar10-image-recognition/workflows/CI%20Pipeline/badge.svg)](https://github.com/rohan727566/cifar10-image-recognition/actions)

> **Final Year Project (2025-2026)**  
> Department of Computer Science and Engineering  
> Pranveer Singh Institute of Technology (PSIT), Kanpur

---

## 📋 Project Overview

An advanced **image recognition system** built using Convolutional Neural Networks (CNN) to classify images from the **CIFAR-10 dataset** into 10 distinct categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

This project implements a complete ML pipeline with:
- **Model Training** with TensorFlow/Keras
- **REST API** for inference (FastAPI)
- **Web UI** for interactive predictions
- **Automated Testing** (pytest with 9 unit/integration tests)
- **CI/CD Pipeline** (GitHub Actions, 100% passing)

**Project ID:** `26_CS_AI_4C_06`

---

## 👥 Team Members

| Name | Student ID | Role |
|------|-----------|------|
| **Rohan Kumar** | 2201641520143 | Lead Developer, DevOps |
| **Rishit** | 2201641520139 | Backend Developer |
| **Shaurya Vardhan Singh** | 2201641520159 | ML Engineer |
| **Vaishnavi Shukla** | 2201641520193 | Frontend Developer |
| **Sudhanshu Awasthi** | 2201641520178 | Testing & Documentation |

**Supervisor:** Mr. Ashish Kumar Singh (Assistant Professor)

---

## 🎯 Features

- ✅ **CNN-based image classification** (10 CIFAR-10 classes)
- ✅ **FastAPI REST API** with automatic Swagger UI documentation
- ✅ **Web-based UI** for real-time image upload and predictions
- ✅ **Training pipeline** with callbacks (ModelCheckpoint, EarlyStopping)
- ✅ **Evaluation metrics** (confusion matrix, classification report)
- ✅ **Automated tests** (9 tests, 48%+ code coverage)
- ✅ **CI/CD pipelines** (GitHub Actions, fully passing)
- ✅ **Production-ready code** (type hints, logging, error handling)
- ✅ **Docker support** (Dockerfile, docker-compose.yml for containerization)
- ✅ **Render deployment** ready (optional)

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.11** or higher
- **Git**
- Virtual environment (recommended)

### 1. Clone Repository
git clone https://github.com/rohan727566/cifar10-image-recognition.git
cd cifar10-image-recognition

### 2. Setup Virtual Environment
Windows
python -m venv .venv
.venv\Scripts\activate

Linux/macOS
python3 -m venv .venv
source .venv/bin/activate

### 3. Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt

### 4. Train the Model (Optional)
Train for 18 epochs (takes ~15-20 minutes)
python -m src.cifar_app.train --epochs 18 --batch-size 32 --save-dir model

Quick test with 1 epoch
python -m src.cifar_app.train --epochs 1 --batch-size 32 --save-dir model

This generates:
- `model/saved_model/` (TensorFlow SavedModel format)
- `model/best_model.keras` (Best checkpoint)
- `model/weights.h5` (Model weights)
- `model/history.json` (Training metrics)

### 5. Run the API Server
python -m src.cifar_app.server

Server starts at: [**http://127.0.0.1:8000**](http://127.0.0.1:8000)

### 6. Open Web UI (in separate terminal)
python -m http.server 5500 --directory web


Access UI at: [**http://localhost:5500**](http://localhost:5500)

Upload an image and get top-3 predictions with confidence scores!

---

## 📦 Installation

### Full Installation (Development)
Clone repo
git clone https://github.com/rohan727566/cifar10-image-recognition.git
cd cifar10-image-recognition

Create virtual environment
python -m venv .venv

Activate (Windows)
.venv\Scripts\activate

OR (Linux/macOS)
source .venv/bin/activate

Install all dependencies
pip install -r requirements.txt


### Installed Packages
| Package | Version | Purpose |
|---------|---------|---------|
| tensorflow | 2.15.0 | Deep learning framework |
| keras | 2.15.0 | Neural network API |
| fastapi | 0.104.1 | Web framework for API |
| uvicorn | 0.24.0 | ASGI server |
| numpy | 1.26.2 | Numerical computing |
| pillow | 10.1.0 | Image processing |
| scikit-learn | 1.3.2 | ML metrics & utilities |
| pytest | 7.4.3 | Testing framework |
| black | 23.11.0 | Code formatting |
| ruff | 0.1.6 | Code linting |

---

## 🧪 Testing

### Run All Tests
pytest -q

Expected output:
9 passed in ~10s


### Run Tests with Verbose Output
pytest tests/ -v --tb=short

### Run with Coverage Report
pytest tests/ --cov=src/cifar_app --cov-report=html

Coverage report generated in htmlcov/index.html

### Test Structure
- **tests/test_model.py** (5 tests)
  - Data preprocessing validation
  - Model architecture verification
  - Prediction shape testing
  
- **tests/test_api.py** (4 tests)
  - Health endpoint
  - Valid image prediction
  - Error handling

**Coverage:** 48%+ across core modules

---

## 🐳 Docker (Optional)

### Build Docker Image
docker build -t cifar10-image-recognition:latest.


### Run Container with docker-compose
docker-compose up -d


Access the API at: [**http://localhost:8000**](http://localhost:8000)

### Stop Container
docker-compose down


---

## 📊 Model Architecture

### CNN Architecture
Input (32×32×3)
↓
[Conv2D(32, 3×3) + Conv2D(32, 3×3) + MaxPool(2×2) + Dropout(0.25)]
↓
[Conv2D(64, 3×3) + Conv2D(64, 3×3) + MaxPool(2×2) + Dropout(0.25)]
↓
Flatten
↓
Dense(512, relu) + Dropout(0.5)
↓
Dense(10, softmax) [Output: 10 classes]

**Total Parameters:** 1,250,858  
**Training:** Adam optimizer, categorical cross-entropy loss  
**Metrics:** Accuracy (training & validation)

### CIFAR-10 Classes
1. airplane
2. automobile
3. bird
4. cat
5. deer
6. dog
7. frog
8. horse
9. ship
10. truck

---

## 📁 Project Structure

cifar10-image-recognition/
├── src/cifar_app/
│ ├── init.py # Package initialization
│ ├── config.py # Centralized configuration
│ ├── preprocess.py # Data loading & preprocessing
│ ├── model.py # CNN architecture & prediction
│ ├── train.py # Training script (CLI)
│ ├── evaluate.py # Evaluation & metrics
│ └── server.py # FastAPI server
├── tests/
│ ├── test_model.py # Unit tests for model
│ └── test_api.py # API endpoint tests
├── web/
│ ├── index.html # Web UI
│ └── static/
│ ├── app.js # Frontend logic
│ └── styles.css # Styling
├── .github/workflows/
│ ├── ci.yml # CI pipeline (testing)
│ └── cd.yml # CD pipeline (deployment)
├── model/ # Generated at runtime
│ ├── saved_model/ # TensorFlow SavedModel
│ ├── best_model.keras # Best checkpoint
│ ├── weights.h5 # Model weights
│ └── history.json # Training metrics
├── docs/plots/ # Generated evaluation plots
│ ├── confusion_matrix.png
│ └── classification_report.json
├── requirements.txt # Dependencies
├── setup.py # Package setup
├── pyproject.toml # Project config (pytest, black, ruff)
├── Dockerfile # Docker configuration
├── docker-compose.yml # Multi-container setup
├── README.md # This file
├── CHANGELOG.md # Version history
├── CONTRIBUTING.md # Development guidelines
└── LICENSE # MIT License

---

## 🔌 API Documentation

### Health Check
curl http://127.0.0.1:8000/health

**Response:**
{"status": "ok"}


### Make Prediction
curl -F "file=@image.png" http://127.0.0.1:8000/predict

**Response:**
{
"predicted_class": 3,
"predicted_label": "cat",
"confidence": 0.9234,
"top3": [
{"class": 3, "label": "cat", "probability": 0.9234},
{"class": 4, "label": "deer", "probability": 0.0512},
{"class": 6, "label": "frog", "probability": 0.0254}
],
"preprocessing_info": "Image resized to 32x32, normalized.",
"inference_time_ms": 156.32
}


### Auto API Docs
Visit [**http://127.0.0.1:8000/docs**](http://127.0.0.1:8000/docs) for interactive Swagger UI

---

## 📈 Training & Evaluation

### Train Model
python -m src.cifar_app.train
--epochs 18
--batch-size 32
--save-dir model
--validation-split 0.2


### Evaluate Model
python -m src.cifar_app.evaluate
--model model/saved_model
--output docs/plots


Generates:
- `docs/plots/confusion_matrix.png` (10×10 heatmap)
- `docs/plots/classification_report.json` (Precision, Recall, F1-score per class)

---

## 🔄 CI/CD Pipeline

### GitHub Actions Workflows

**CI Pipeline** (ci.yml)
- Triggered: Every push to `develop` or `main`
- Steps:
  - Python 3.11 setup
  - Install dependencies
  - Linting (ruff, black)
  - Run 9 tests with coverage
  - Status: ✅ All checks passing

**CD Pipeline** (cd.yml)
- Triggered: Push to `main` branch
- Deploys to Render (optional, requires webhook)

---

## 🌐 Deployment to Render (Optional)

### Prerequisites
- Render account (https://render.com)
- GitHub account with this repo

### Steps
1. Log in to Render Dashboard
2. Click "New +" → "Web Service"
3. Select GitHub repo: `cifar10-image-recognition`
4. Configure:
   - **Name:** `cifar10-image-recognition`
   - **Branch:** `develop` or `main`
   - **Build Command:** `pip install -r requirements.txt && python -m src.cifar_app.train --epochs 5 --save-dir model`
   - **Start Command:** `uvicorn src.cifar_app.server:app --host 0.0.0.0 --port 10000`
5. Click "Create Web Service"
6. Wait ~5-10 minutes for deployment
7. Access live URL: `https://YOUR-APP-NAME.onrender.com`

---

## 📝 Command Cheatsheet

Setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

Training
python -m src.cifar_app.train --epochs 18 --save-dir model

Evaluation
python -m src.cifar_app.evaluate --model model/saved_model --output docs/plots

Server
python -m src.cifar_app.server

Web UI
python -m http.server 5500 --directory web

Testing
pytest -q
pytest tests/ -v
pytest tests/ --cov=src/cifar_app

Git
git add .
git commit -m "message"
git push origin develop


---

## 📚 Documentation

- **README.md** — This file (overview, setup, usage)
- **CHANGELOG.md** — Version history & milestones
- **CONTRIBUTING.md** — Development guidelines
- **LICENSE** — MIT License (all team members)

---

## 🎓 Academic Context

This project is a **Final Year Project (FYP)** submitted to:
- **Institution:** Pranveer Singh Institute of Technology, Kanpur
- **Department:** Computer Science and Engineering
- **Project ID:** 26_CS_AI_4C_06
- **Academic Year:** 2025-2026
- **Supervisor:** Mr. Ashish Kumar Singh (Assistant Professor)

**Deliverables:**
- ✅ Functional image classification system
- ✅ REST API for inference
- ✅ Web-based UI
- ✅ Automated testing suite
- ✅ CI/CD pipeline
- ✅ Complete documentation
- ✅ Production-ready code

---

## 🔗 Links

- **GitHub Repository:** https://github.com/rohan727566/cifar10-image-recognition
- **CIFAR-10 Dataset:** https://www.cs.toronto.edu/~kriz/cifar.html
- **TensorFlow Docs:** https://tensorflow.org/
- **FastAPI Docs:** https://fastapi.tiangolo.com/
- **Render Docs:** https://render.com/docs

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

**Copyright © 2025** Rohan Kumar, Rishit, Shaurya Vardhan Singh, Vaishnavi Shukla, Sudhanshu Awasthi

---

## 🙏 Acknowledgments

- **Pranveer Singh Institute of Technology, Kanpur** for academic support
- **CIFAR-10 Dataset Creators** for the high-quality dataset
- **TensorFlow & Keras Communities** for excellent deep learning tools
- **FastAPI** for the modern web framework
- **GitHub & Render** for CI/CD and deployment infrastructure

---

## ❓ FAQ

**Q: How do I train with my own data?**  
A: Modify `preprocess.py` to load your dataset instead of CIFAR-10, and run the training script.

**Q: Can I use this model for real-world images?**  
A: The CIFAR-10 model is trained on 32×32 images. It will work on different sizes (resized), but accuracy may vary.

**Q: How do I improve model accuracy?**  
A: Increase epochs, try data augmentation, adjust hyperparameters in `config.py`.

**Q: Is GPU required?**  
A: No, but training will be faster with GPU (NVIDIA CUDA or Apple Metal).

---

**Last Updated:** November 30, 2025  
**Status:** ✅ Production Ready

---
