import sys
import types
import traceback

# First, patch all potentially problematic OpenCV modules
def patch_cv2_modules():
    """
    Create mock modules to prevent circular import errors in OpenCV.
    This function creates dummy modules and classes for the problematic parts of OpenCV.
    """
    # Create mock modules
    for module_name in [
        'cv2.gapi', 
        'cv2.typing', 
        'cv2.dnn', 
        'cv2.gapi.wip', 
        'cv2.gapi.wip.draw', 
        'cv2.gapi_wip_gst_GStreamerPipeline'
    ]:
        if module_name not in sys.modules:
            dummy_module = types.ModuleType(module_name)
            sys.modules[module_name] = dummy_module
    
    # Add missing attributes to the gapi.wip.draw module
    draw_module = sys.modules.get('cv2.gapi.wip.draw')
    if draw_module:
        class DummyClass:
            def __getattr__(self, name):
                return DummyClass()
            
            def __call__(self, *args, **kwargs):
                return DummyClass()
        
        # Add all the missing attributes mentioned in the error
        draw_module.Text = DummyClass()
        draw_module.Circle = DummyClass()
        draw_module.Image = DummyClass()
        draw_module.Line = DummyClass()
        draw_module.Rect = DummyClass()
        draw_module.Mosaic = DummyClass()
        draw_module.Poly = DummyClass()
    
    # Handle gapi module
    gapi_module = sys.modules.get('cv2.gapi')
    if gapi_module:
        class DummyClass:
            def __getattr__(self, name):
                return DummyClass()
            
            def __call__(self, *args, **kwargs):
                return DummyClass()
        
        # Add missing attributes
        gapi_module.wip = DummyClass()
        setattr(gapi_module.wip, 'draw', DummyClass())
        setattr(gapi_module.wip, 'GStreamerPipeline', DummyClass())
    
    print("OpenCV modules patched to prevent circular imports")

# Apply the patch
patch_cv2_modules()

# Import BEFORE cv2
from ultralytics import YOLO

# Now safely import OpenCV
import cv2

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import os
import pytesseract
from PIL import Image
import io
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
model_lock = threading.Lock()

@app.get("/version")
async def version():
    """Get environment version information"""
    import torch
    import platform
    import ultralytics
    
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "ultralytics_version": ultralytics.__version__,
        "opencv_version": cv2.__version__
    }

def get_model():
    global model
    with model_lock:
        if model is None:
            try:
                # Change loading technique to avoid segmentation fault
                print("Loading YOLO model differently...")
                import torch
                
                # Ensure model loading is done with error handling
                try:
                    # Apply necessary device and weight settings
                    # This avoids the classic error with torch.load
                    model = YOLO('best.pt')
                    print("YOLO model loaded successfully")
                except Exception as e:
                    print(f"Failed YOLO main load: {e}")
                    print(traceback.format_exc())
                    raise
            except Exception as e:
                print(f"Error loading YOLO model: {e}")
                print(traceback.format_exc())
                raise
    return model

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
    # Convert numpy array to PIL Image for compatibility
    pil_image = Image.fromarray(image)
    text = pytesseract.image_to_string(pil_image, lang='eng')
    return text.strip()

def extract_text_easyocr(image):
    """Extract text using EasyOCR"""
    # Import here to avoid circular imports
    try:
        import easyocr
        reader = easyocr.Reader(['en'])
        results = reader.readtext(image)
        return ' '.join([text for _, text, conf in results if conf > 0.3])
    except Exception as e:
        print(f"EasyOCR error: {e}")
        return ""

def extract_text_combined(image):
    """Combine both OCR methods for better results"""
    text_tesseract = extract_text_tesseract(image)
    
    try:
        text_easyocr = extract_text_easyocr(image)
        # Use EasyOCR result if Tesseract result is empty or contains unwanted characters
        text = text_easyocr if not text_tesseract.strip() else text_tesseract
    except:
        # Fall back to just Tesseract if EasyOCR fails
        text = text_tesseract
    
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
        
        print(f"File saved to {filepath}")
        
        # Run YOLOv8 inference with lazy-loaded model
        try:
            # Try detection with error handling
            yolo_model = get_model()
            print("Starting inference...")
            results = yolo_model(filepath, verbose=False)[0]
            print("Inference completed successfully")
        except Exception as e:
            print(f"Error during model inference: {e}")
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Model inference error: {str(e)}")
        
        # Process detections
        try:
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
                
                print(f"Processing detection: {class_name} with confidence {conf}")
                
                # Crop the region
                cropped = crop_box(image, bbox)
                
                # Preprocess the cropped image
                processed = preprocess_for_ocr(cropped)
                
                # Extract text
                try:
                    text = extract_text_combined(processed)
                except Exception as ocr_error:
                    print(f"OCR error: {ocr_error}")
                    text = "OCR failed"
                
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
            print("Drawing bounding boxes")
            annotated_image = draw_boxes(image, predictions)
            
            # Save annotated image
            annotated_path = os.path.join(UPLOAD_FOLDER, f"annotated_{file.filename or 'image'}{file_extension}")
            cv2.imwrite(annotated_path, annotated_image)
            
            # Convert image to base64 for response
            _, buffer = cv2.imencode('.jpg', annotated_image)
            annotated_b64 = base64.b64encode(buffer).decode('utf-8')
            
            print("Returning results")
            
            # Clean up original file
            os.remove(filepath)
            
            return {
                'extracted_info': extracted_info,
                'annotated_image': annotated_b64
            }
        except Exception as e:
            print(f"Processing error: {e}")
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    except Exception as e:
        # Error handling
        print(f"General error: {e}")
        print(traceback.format_exc())
        if os.path.exists(filepath):
            os.remove(filepath)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

# For local development
if __name__ == '__main__':
    # Run the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get('PORT', 8000)))