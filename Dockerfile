FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with compatible torchvision
RUN pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model files
COPY . .

# Create upload directory
RUN mkdir -p uploads

# Environment variables
ENV PYTHONUNBUFFERED=1

# Expose the port
EXPOSE 8000

# Run with Uvicorn
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --log-level debug