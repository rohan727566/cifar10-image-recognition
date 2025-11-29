"""Setup script for cifar10-image-recognition package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cifar10-image-recognition",
    version="1.0.0",
    author="Rohan Kumar, Rishit, Shaurya Vardhan Singh, Vaishnavi Shukla, Sudhanshu Awasthi",
    author_email="rohan727566@users.noreply.github.com",
    description="Image Recognition System using CIFAR-10 Dataset with CNN",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rohan727566/cifar10-image-recognition",
    project_urls={
        "Bug Tracker": "https://github.com/rohan727566/cifar10-image-recognition/issues",
        "Documentation": "https://github.com/rohan727566/cifar10-image-recognition#readme",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.11",
    install_requires=[
        "tensorflow==2.15.0",
        "keras==2.15.0",
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0",
        "gunicorn==21.2.0",
        "python-multipart==0.0.6",
        "numpy==1.26.2",
        "matplotlib==3.8.2",
        "seaborn==0.13.0",
        "pillow==10.1.0",
        "scikit-learn==1.3.2",
        "python-dotenv==1.0.0",
        "click==8.1.7",
    ],
    extras_require={
        "dev": [
            "pytest==7.4.3",
            "pytest-cov==4.1.0",
            "httpx==0.25.2",
            "ruff==0.1.6",
            "black==23.11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cifar-train=cifar_app.train:main",
            "cifar-server=cifar_app.server:main",
            "cifar-evaluate=cifar_app.evaluate:main",
        ],
    },
)
