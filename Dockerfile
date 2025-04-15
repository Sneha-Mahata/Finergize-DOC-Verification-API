FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with compatible version
RUN pip install --no-cache-dir torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torchvision --index-url https://download.pytorch.org/whl/cpu

# Install required Python packages
RUN pip install --no-cache-dir fastapi==0.95.1 uvicorn==0.22.0 \
    python-multipart==0.0.6 numpy==1.23.5 Pillow==9.5.0 \
    pytesseract==0.3.9 ultralytics==8.3.108

# Copy application code
COPY robust_main.py main.py

# Create required directories
RUN mkdir -p uploads results

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Copy model file
COPY best.pt .

# Expose the port
EXPOSE 8000

# Run with timeout settings
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --log-level debug --timeout-keep-alive 120