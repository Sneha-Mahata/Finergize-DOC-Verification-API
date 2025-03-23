FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Install dependencies one by one to avoid resolution conflicts
RUN pip install --no-cache-dir Flask==2.0.1 \
    && pip install --no-cache-dir numpy==1.22.0 \
    && pip install --no-cache-dir pytesseract==0.3.9 \
    && pip install --no-cache-dir Pillow==9.0.1 \
    && pip install --no-cache-dir gunicorn==20.1.0 \
    && pip install --no-cache-dir werkzeug==2.0.1 \
    && pip install --no-cache-dir opencv-python-headless==4.5.5.64 \
    && pip install --no-cache-dir easyocr==1.6.2 \
    && pip install --no-cache-dir ultralytics==8.0.20

# Copy application code and model files
COPY app.py .
COPY best.pt .
COPY data.yaml .

# Create upload directory
RUN mkdir -p uploads

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
CMD gunicorn --bind 0.0.0.0:$PORT app:app
