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

# Install PyTorch separately first (no torchvision to avoid conflicts)
RUN pip install --no-cache-dir torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Install the cv2_patch.py first
COPY cv2_patch.py .
COPY main.py .
COPY requirements.txt .

# Install specific version of opencv first (to avoid circular import issues)
RUN pip install --no-cache-dir opencv-python-headless==4.7.0.72

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else
COPY . .

# Create upload directory
RUN mkdir -p uploads

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Expose the port
EXPOSE 8000

# Run with increased max request size and timeout
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --log-level debug --timeout-keep-alive 120