FROM python:3.9-slim

WORKDIR /app

# Install system dependencies 
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch first (for better layer caching)
RUN pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu

# Pre-install opencv separately to handle its dependencies better
RUN pip install --no-cache-dir opencv-python-headless==4.5.1.48

# Install Python dependencies separately for better caching
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

# Run FastAPI with Uvicorn
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}