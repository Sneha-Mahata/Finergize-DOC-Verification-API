FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies in the correct order
RUN pip install --no-cache-dir numpy==1.23.5
# Use opencv-python instead of opencv-python-headless
RUN pip install --no-cache-dir opencv-python==4.5.1.48
RUN pip install --no-cache-dir Flask==2.0.1 \
    werkzeug==2.0.1 \
    gunicorn==20.1.0 \
    Pillow==9.0.1 \
    pytesseract==0.3.9 \
    easyocr==1.6.2 \
    ultralytics==8.0.20

# Copy application code and model files
COPY . .

# Create upload directory
RUN mkdir -p uploads

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
CMD gunicorn --bind 0.0.0.0:$PORT app:app