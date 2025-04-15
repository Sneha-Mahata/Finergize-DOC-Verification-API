from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os
import easyocr
import pytesseract
from PIL import Image
import io
from ultralytics import YOLO
import base64
import threading
import uvicorn
from typing import Dict, List, Any, Optional

# Create FastAPI app
app = FastAPI(
    title="Document Verification API",
    description="API for detecting and extracting text from identity documents using YOLO and OCR",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create upload directory
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for lazy loading
model = None
reader = None
model_lock = threading.Lock()
reader_lock = threading.Lock()

def get_model():
    global model
    with model_lock:
        if model is None:
            try:
                # Load the model directly using YOLO
                model = YOLO('best.pt')
            except Exception as e:
                print(f"Error loading YOLO model: {e}")
                raise
    return model

def get_reader():
    global reader
    with reader_lock:
        if reader is None:
            try:
                reader = easyocr.Reader(['en'])
            except Exception as e:
                print(f"Error initializing EasyOCR: {e}")
                raise
    return reader

def crop_box(image, bbox):
    """Crop image with bounding box"""
    xmin, ymin, xmax, ymax = map(int, bbox)
    return image[ymin:ymax, xmin:xmax]

def preprocess_for_ocr(image):
    """Preprocess image for OCR"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to get better text recognition
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Add padding
    padded = cv2.copyMakeBorder(thresh, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
    
    return padded

def extract_text_tesseract(image):
    """Extract text using Tesseract OCR"""
    text = pytesseract.image_to_string(image, lang='eng')
    return text.strip()

def extract_text_easyocr(image):
    """Extract text using EasyOCR"""
    results = get_reader().readtext(image)
    return ' '.join([text for _, text, conf in results if conf > 0.3])

def extract_text_combined(image):
    """Combine both OCR methods for better results"""
    text_tesseract = extract_text_tesseract(image)
    text_easyocr = extract_text_easyocr(image)
    
    # Use EasyOCR result if Tesseract result is empty or contains unwanted characters
    text = text_easyocr if not text_tesseract.strip() else text_tesseract
    
    return text.strip()

def draw_boxes(image, predictions):
    """Draw bounding boxes on the image"""
    output = image.copy()
    for pred in predictions:
        bbox = pred['bbox']
        label = f"{pred['class_name']}: {pred['text']}"
        
        # Draw rectangle
        cv2.rectangle(output, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        
        # Add label
        cv2.putText(output, label, (int(bbox[0]), int(bbox[1] - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return output

@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {
        "status": "active",
        "message": "Document Verification API is running. Use POST /predict to analyze images."
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Analyze an image for document fields and extract text
    
    Args:
        file: Image file to analyze
    
    Returns:
        JSON response with extracted information and annotated image
    """
    # Validate the file
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Save the uploaded file
    file_extension = os.path.splitext(file.filename)[1] if file.filename else ".jpg"
    filepath = os.path.join(UPLOAD_FOLDER, f"upload_{file.filename or 'image'}{file_extension}")
    
    try:
        # Read file content
        contents = await file.read()
        with open(filepath, "wb") as f:
            f.write(contents)
        
        # Run YOLOv8 inference with lazy-loaded model
        results = get_model()(filepath)[0]
        
        # Process detections
        image = cv2.imread(filepath)
        predictions = []
        extracted_info = {}
        
        for i, box in enumerate(results.boxes):
            bbox = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = results.names[cls_id]
            
            # Skip low confidence detections
            if conf < 0.5:
                continue
            
            # Crop the region
            cropped = crop_box(image, bbox)
            
            # Preprocess the cropped image
            processed = preprocess_for_ocr(cropped)
            
            # Extract text
            text = extract_text_combined(processed)
            
            pred_info = {
                'class_name': class_name,
                'confidence': conf,
                'bbox': bbox,
                'text': text
            }
            
            predictions.append(pred_info)
            extracted_info[class_name] = {
                'text': text,
                'confidence': conf,
                'bbox': bbox
            }
        
        # Draw bounding boxes
        annotated_image = draw_boxes(image, predictions)
        
        # Save annotated image
        annotated_path = os.path.join(UPLOAD_FOLDER, f"annotated_{file.filename or 'image'}{file_extension}")
        cv2.imwrite(annotated_path, annotated_image)
        
        # Convert image to base64 for response
        _, buffer = cv2.imencode('.jpg', annotated_image)
        annotated_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Clean up original file
        os.remove(filepath)
        
        return {
            'extracted_info': extracted_info,
            'annotated_image': annotated_b64
        }
    
    except Exception as e:
        # Error handling
        if os.path.exists(filepath):
            os.remove(filepath)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

# For local development
if __name__ == '__main__':
    # Pre-load model
    print("Preloading models...")
    try:
        get_model()
        get_reader()
        print("Models loaded successfully")
    except Exception as e:
        print(f"Error preloading models: {e}")
    
    # Run the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get('PORT', 8000)))