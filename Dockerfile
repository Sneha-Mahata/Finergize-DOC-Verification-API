FROM continuumio/miniconda3:latest

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create and activate a conda environment with OpenCV pre-installed
RUN conda create -n app_env python=3.9 opencv=4.5.1 -c conda-forge && \
    echo "conda activate app_env" >> ~/.bashrc

# Set the shell to bash for conda activation
SHELL ["/bin/bash", "--login", "-c"]

# Install Python dependencies in the conda environment
RUN conda activate app_env && \
    pip install flask==2.0.1 \
    werkzeug==2.0.1 \
    gunicorn==20.1.0 \
    pillow==9.0.1 \
    pytesseract==0.3.9 \
    easyocr==1.6.2 \
    ultralytics==8.0.20

# Copy application code and model files
COPY . .

# Create upload directory
RUN mkdir -p uploads

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application with the conda environment activated
CMD conda run -n app_env python -c "import cv2_patch; import gunicorn.app.wsgiapp; gunicorn.app.wsgiapp.run()" --bind 0.0.0.0:$PORT app:app