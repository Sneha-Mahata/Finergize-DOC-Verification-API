FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and patch files first
COPY requirements.txt cv2_patch.py ./

# Install dependencies in one step to reduce Docker layers
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model files
COPY . .

# Create upload directory
RUN mkdir -p uploads

# Set environment variables
ENV PYTHONUNBUFFERED=1 
ENV PYTHONPATH=/app

# Run the application
CMD gunicorn --bind 0.0.0.0:$PORT app:app