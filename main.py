from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import os
import subprocess
import sys
import json
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import uvicorn
import traceback
from typing import Dict, List, Any, Optional
import uuid
import time

# Create FastAPI app
app = FastAPI(
    title="Document Verification API",
    description="API for detecting and extracting text from identity documents",
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
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def extract_text_tesseract(image_path):
    """Extract text using Tesseract OCR"""
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, lang='eng')
        return text.strip()
    except Exception as e:
        print(f"Tesseract error: {e}")
        return ""

def draw_boxes_pil(image_path, predictions):
    """Draw bounding boxes using PIL instead of OpenCV"""
    try:
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        
        # Try to load a font, use default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except:
            font = ImageFont.load_default()
        
        for pred in predictions:
            bbox = pred['bbox']
            label = f"{pred['class_name']}: {pred['text']}"
            
            # Draw rectangle
            draw.rectangle(
                [(bbox[0], bbox[1]), (bbox[2], bbox[3])], 
                outline="green", 
                width=2
            )
            
            # Add label
            draw.text(
                (bbox[0], bbox[1] - 15), 
                label, 
                fill="green",
                font=font
            )
        
        output_path = image_path.replace('uploads', 'results')
        image.save(output_path)
        return output_path
    except Exception as e:
        print(f"Error drawing boxes: {e}")
        print(traceback.format_exc())
        return None

def run_yolo_detection(image_path, output_path):
    """Run YOLO detection using the yolo command-line tool"""
    try:
        # Run yolo command as a subprocess to avoid memory issues
        cmd = [
            "python3", "-c",
            f"""
import torch
from ultralytics import YOLO
import json
import numpy as np

# Custom encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

# Load model
try:
    model = YOLO('best.pt')
    
    # Run detection
    results = model('{image_path}', verbose=False)[0]
    
    # Process results
    detections = []
    for i, box in enumerate(results.boxes):
        bbox = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        class_name = results.names[cls_id]
        
        # Skip low confidence detections
        if conf < 0.5:
            continue
            
        detections.append({{
            'class_name': class_name,
            'confidence': conf,
            'bbox': bbox.tolist()
        }})
    
    # Save as JSON
    with open('{output_path}', 'w') as f:
        json.dump(detections, f, cls=NumpyEncoder)
    
    print('Detection completed successfully')
except Exception as e:
    import traceback
    with open('{output_path}.error', 'w') as f:
        f.write(str(e) + '\\n' + traceback.format_exc())
    print(f'Error: {{e}}')
"""
        ]
        
        # Run the command and capture output
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(f"YOLO detection output: {result.stdout}")
        print(f"YOLO detection errors: {result.stderr}")
        
        # Check if detection was successful
        if not os.path.exists(output_path):
            if os.path.exists(f"{output_path}.error"):
                with open(f"{output_path}.error", 'r') as f:
                    error = f.read()
                print(f"YOLO detection error: {error}")
                raise Exception(f"Model error: {error}")
            raise Exception("YOLO detection failed")
            
        # Load the results
        with open(output_path, 'r') as f:
            detections = json.load(f)
            
        return detections
    except Exception as e:
        print(f"Error running YOLO detection: {e}")
        print(traceback.format_exc())
        raise

def process_image(filepath, json_path):
    """Process an image with YOLO detection and OCR"""
    try:
        # Run YOLO detection
        detections = run_yolo_detection(filepath, json_path)
        
        # Extract text from detections
        for detection in detections:
            # Get coordinates
            bbox = detection['bbox']
            
            # Crop the image
            try:
                image = Image.open(filepath)
                cropped = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                
                # Save cropped image temporarily
                temp_crop_path = f"{filepath}_{detection['class_name']}_crop.jpg"
                cropped.save(temp_crop_path)
                
                # Extract text using Tesseract
                text = extract_text_tesseract(temp_crop_path)
                detection['text'] = text
                
                # Clean up temporary file
                if os.path.exists(temp_crop_path):
                    os.remove(temp_crop_path)
            except Exception as e:
                print(f"Error extracting text for {detection['class_name']}: {e}")
                detection['text'] = "Text extraction failed"
        
        # Draw bounding boxes
        annotated_path = draw_boxes_pil(filepath, detections)
        
        # Create extracted_info from detections
        extracted_info = {}
        for detection in detections:
            extracted_info[detection['class_name']] = {
                'text': detection['text'],
                'confidence': detection['confidence'],
                'bbox': detection['bbox']
            }
        
        # Convert annotated image to base64
        if annotated_path and os.path.exists(annotated_path):
            with open(annotated_path, "rb") as img_file:
                annotated_b64 = base64.b64encode(img_file.read()).decode('utf-8')
        else:
            annotated_b64 = None
        
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)
        if os.path.exists(json_path):
            os.remove(json_path)
        if annotated_path and os.path.exists(annotated_path):
            os.remove(annotated_path)
        
        return {
            'extracted_info': extracted_info,
            'annotated_image': annotated_b64
        }
    except Exception as e:
        print(f"Error processing image: {e}")
        print(traceback.format_exc())
        # Clean up files
        for f in [filepath, json_path]:
            if os.path.exists(f):
                os.remove(f)
        raise

@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {
        "status": "active",
        "message": "Document Verification API is running. Use POST /predict to analyze images."
    }

@app.get("/version")
async def version():
    """Get version information"""
    import platform
    
    # Check if YOLO is available
    try:
        import ultralytics
        ultralytics_version = ultralytics.__version__
    except:
        ultralytics_version = "Not available"
    
    # Check if PyTorch is available
    try:
        import torch
        torch_version = torch.__version__
    except:
        torch_version = "Not available"
    
    return {
        "python_version": platform.python_version(),
        "ultralytics_version": ultralytics_version,
        "torch_version": torch_version
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...), background_tasks: BackgroundTasks):
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
    
    # Generate unique filename
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1] if file.filename else ".jpg"
    filepath = os.path.join(UPLOAD_FOLDER, f"{file_id}{file_extension}")
    json_path = os.path.join(RESULTS_FOLDER, f"{file_id}.json")
    
    try:
        # Read file content
        contents = await file.read()
        with open(filepath, "wb") as f:
            f.write(contents)
        
        print(f"File saved to {filepath}")
        
        # Process the image
        result = process_image(filepath, json_path)
        
        return result
    except Exception as e:
        # Error handling
        print(f"Error processing request: {e}")
        print(traceback.format_exc())
        # Clean up files
        for f in [filepath, json_path]:
            if os.path.exists(f):
                os.remove(f)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

# For local development
if __name__ == '__main__':
    # Run the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get('PORT', 8000)))