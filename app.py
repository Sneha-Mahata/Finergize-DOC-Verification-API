import sys
import types
import os

# Set a higher timeout for gunicorn
os.environ["GUNICORN_CMD_ARGS"] = "--timeout=300"

# Create mock objects with all necessary attributes
class DummyObject:
    def __getattr__(self, name):
        return DummyObject()
    
    def __call__(self, *args, **kwargs):
        return DummyObject()

# Mock the problematic modules
sys.modules['cv2.gapi'] = DummyObject()
sys.modules['cv2.typing'] = DummyObject()
sys.modules['cv2.gapi.wip'] = DummyObject()
sys.modules['cv2.gapi.wip.draw'] = DummyObject()

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
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

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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
                # Import the patch function if not already done in app_wrapper
                try:
                    from patch_ultralytics import patch_ultralytics_modules
                    patch_ultralytics_modules()
                except ImportError:
                    pass
                
                # Now load the model
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

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "status": "active",
        "message": "Document Verification API is running. Use POST /predict to analyze images."
    })

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save the uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
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
        annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], f"annotated_{filename}")
        cv2.imwrite(annotated_path, annotated_image)
        
        # Convert image to base64 for response
        _, buffer = cv2.imencode('.jpg', annotated_image)
        annotated_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Clean up original file
        os.remove(filepath)
        
        return jsonify({
            'extracted_info': extracted_info,
            'annotated_image': annotated_b64
        })
    
    except Exception as e:
        # Error handling
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({
            'error': f'Processing error: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Pre-load model in development for faster response
    if os.environ.get('FLASK_ENV') == 'development':
        print("Preloading models...")
        get_model()
        get_reader()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))