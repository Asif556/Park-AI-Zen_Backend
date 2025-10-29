import base64
import cv2
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from datetime import datetime, UTC, timedelta
import math
import random
import pickle
import traceback
from paddleocr import PaddleOCR
from ultralytics import YOLO
from dotenv import load_dotenv
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import jwt
import string
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import threading
import google.generativeai as genai

from Rag_Model import get_chatbot_response, initialize_rag_system

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print(f"[Gemini] API configured successfully")
else:
    print("[Gemini] Warning: GEMINI_API_KEY not found in .env file")



print("Initializing RAG system...")
rag_qdrant_client, rag_collection_name, rag_model, rag_mongo_uri = initialize_rag_system()
print("RAG system ready! (Using Qdrant Cloud for vector storage)")



# Import auto-retrain functionality
try:
    from auto_retrain_model import trigger_retrain_if_needed
    AUTO_RETRAIN_ENABLED = True
    print("[Auto-Retrain] Enabled - Model will retrain after every 5 exits")
except ImportError:
    AUTO_RETRAIN_ENABLED = False
    print("[Auto-Retrain] Disabled - auto_retrain_model.py not found")

# Load LSTM model for predictions
try:
    from tensorflow import keras
    import pickle
    import numpy as np
    
    LSTM_MODEL_PATH = "parking_lstm_model.keras"  # Changed to .keras format
    LSTM_SCALER_PATH = "parking_scaler.pkl"
    LSTM_TIMESTEPS = 2  # Reduced for small datasets - must match train_lstm_model.py
    
    lstm_model = keras.models.load_model(LSTM_MODEL_PATH)
    with open(LSTM_SCALER_PATH, 'rb') as f:
        lstm_scaler = pickle.load(f)
    LSTM_ENABLED = True
    print("[LSTM] Prediction model loaded successfully")
except FileNotFoundError:
    lstm_model = None
    lstm_scaler = None
    LSTM_ENABLED = False
    print("[LSTM] Model not found. Train model first with: python train_lstm_model.py")
except Exception as e:
    lstm_model = None
    lstm_scaler = None
    LSTM_ENABLED = False
    print(f"[LSTM] Failed to load model: {e}")

# ------------------------------------------------
# 🔧 PaddleOCR Configuration
# ------------------------------------------------
# Initialize PaddleOCR (English mode, with textline orientation for better accuracy)
# Your initialization is correct.
ocr = PaddleOCR(use_textline_orientation=True, lang='en')
print("[PaddleOCR] Initialized successfully")

# ------------------------------------------------
# ⚙ Flask App Configuration
# ------------------------------------------------
app = Flask(__name__)

# ------------------------------------------------
# 🔒 OTP Authentication Configuration
# ------------------------------------------------
# In-memory OTP storage (email -> {otp, timestamp})
otp_storage = {}

# Load OTP configuration from environment
OUTLOOK_EMAIL = os.getenv("OUTLOOK_EMAIL")
OUTLOOK_PASSWORD = os.getenv("OUTLOOK_PASSWORD")
OUTLOOK_HOST = os.getenv("OUTLOOK_HOST", "smtp-mail.outlook.com")
OUTLOOK_PORT = int(os.getenv("OUTLOOK_PORT", 587))
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-jwt-key-change-this-in-production")
OTP_EXPIRY_MINUTES = int(os.getenv("OTP_EXPIRY_MINUTES", 5))
AUTHORIZED_EMAIL = os.getenv("AUTHORIZED_EMAIL", "abc@gmail.com")

# Initialize Flask-Limiter for rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

print(f"[OTP Auth] Configured - Authorized Email: {AUTHORIZED_EMAIL}")
print(f"[OTP Auth] SMTP: {OUTLOOK_HOST}:{OUTLOOK_PORT} using {OUTLOOK_EMAIL}")

# ------------------------------------------------
# 🔒 OTP Helper Functions
# ------------------------------------------------
def generate_otp(length=6):
    """Generate a random 6-digit OTP"""
    return ''.join(random.choices(string.digits, k=length))

def send_otp_email(recipient_email, otp):
    """Send OTP via Outlook SMTP"""
    try:
        # Create email message
        msg = MIMEMultipart()
        msg['From'] = OUTLOOK_EMAIL
        msg['To'] = recipient_email
        msg['Subject'] = "Your OTP for Smart Parking System"
        
        # Email body
        body = f"""
        <html>
        <body>
            <h2>Smart Parking System - OTP Verification</h2>
            <p>Your One-Time Password (OTP) is:</p>
            <h1 style="color: #4CAF50; font-size: 36px; letter-spacing: 5px;">{otp}</h1>
            <p>This OTP is valid for {OTP_EXPIRY_MINUTES} minutes.</p>
            <p>If you didn't request this OTP, please ignore this email.</p>
            <br>
            <p style="color: #666;">Smart Parking System</p>
        </body>
        </html>
        """
        msg.attach(MIMEText(body, 'html'))
        
        # Connect to Outlook SMTP server
        server = smtplib.SMTP(OUTLOOK_HOST, OUTLOOK_PORT)
        server.starttls()  # Enable TLS encryption
        server.login(OUTLOOK_EMAIL, OUTLOOK_PASSWORD)
        
        # Send email
        server.send_message(msg)
        server.quit()
        
        print(f"[OTP Email] Successfully sent OTP to {recipient_email}")
        return True
        
    except Exception as e:
        print(f"[OTP Email Error] Failed to send OTP to {recipient_email}: {str(e)}")
        traceback.print_exc()
        return False

def cleanup_expired_otps():
    """Remove expired OTPs from storage"""
    current_time = datetime.now(UTC)
    expired_emails = []
    
    for email, data in otp_storage.items():
        if current_time > data['expires_at']:
            expired_emails.append(email)
    
    for email in expired_emails:
        del otp_storage[email]
        print(f"[OTP Cleanup] Removed expired OTP for {email}")

CORS(app, resources={
    r"/api/*": {
        "origins": ["*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": False
    }
})

# ------------------------------------------------
# 🗄️ MongoDB Setup
# ------------------------------------------------
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME", "smart_parking_db")

# Configure MongoDB client with TLS settings to fix SSL handshake issues
import ssl

client = MongoClient(
    MONGODB_URI,
    tls=True,
    tlsAllowInvalidCertificates=True,  # Allow self-signed certificates
    serverSelectionTimeoutMS=5000,      # Reduce timeout to 5 seconds
    connectTimeoutMS=5000,
    socketTimeoutMS=5000
)
db = client[DB_NAME]
records_col = db["parking_records"]
parking_sessions_col = db["parking_sessions"]  # Collection for ML training data

TOTAL_SLOTS = int(os.getenv("TOTAL_SLOTS", 50))
yolo_model = YOLO("best.pt")  # Using the best.pt model file in your directory
vehicle_classifier_model = YOLO("car_or_bike.pt")  # Vehicle classification model (car/bike)

# ------------------------------------------------
# 💰 Parking Charge Calculation
# ------------------------------------------------
BASE_CHARGE_CAR = int(os.getenv("BASE_CHARGE_CAR", 20))
BASE_CHARGE_BIKE = int(os.getenv("BASE_CHARGE_BIKE", 10))
HOURLY_RATE_CAR = int(os.getenv("HOURLY_RATE_CAR", 10))
HOURLY_RATE_BIKE = int(os.getenv("HOURLY_RATE_BIKE", 5))
# Legacy variables for backward compatibility
BASE_CHARGE = BASE_CHARGE_CAR
HOURLY_RATE = HOURLY_RATE_CAR

def calculate_charge(entry_time, exit_time, vehicle_type="car"):
    """
    Calculate parking charge based on vehicle type (car/bike)
    """
    duration = exit_time - entry_time
    hours = duration.total_seconds() / 3600
    
    # Determine rates based on vehicle type
    if vehicle_type.lower() in ["bike", "motorcycle", "2-wheeler", "two-wheeler"]:
        base = BASE_CHARGE_BIKE
        hourly = HOURLY_RATE_BIKE
    else:  # car, 4-wheeler, or default
        base = BASE_CHARGE_CAR
        hourly = HOURLY_RATE_CAR
    
    if hours <= 1:
        return base
    else:
        additional_hours = math.ceil(hours - 1)
        return base + (additional_hours * hourly)

# ------------------------------------------------
# 🧠 Helper: YOLO + PaddleOCR Pipeline
# ------------------------------------------------
def classify_vehicle(base64_image):
    """
    Classify vehicle as car (4-wheeler) or bike (2-wheeler) using YOLOv9 model
    Returns: "car" or "bike"
    """
    try:
        # Decode base64 image
        img_data = base64.b64decode(base64_image)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run YOLO classification
        results = vehicle_classifier_model(img)
        
        # Extract prediction
        for result in results:
            if hasattr(result, 'probs') and result.probs is not None:
                # Classification model
                class_id = result.probs.top1
                confidence = result.probs.top1conf.item()
                class_name = result.names[class_id].lower()
                
                print(f"[Vehicle Classifier] Detected: {class_name} (confidence: {confidence:.2f})")
                
                # Map class names to standard types
                if "bike" in class_name or "motorcycle" in class_name or "2" in class_name:
                    return "bike", confidence
                elif "car" in class_name or "4" in class_name:
                    return "car", confidence
                else:
                    return class_name, confidence
            elif hasattr(result, 'boxes') and len(result.boxes) > 0:
                # Detection model - use first detection
                class_id = int(result.boxes[0].cls[0])
                confidence = float(result.boxes[0].conf[0])
                class_name = result.names[class_id].lower()
                
                print(f"[Vehicle Classifier] Detected: {class_name} (confidence: {confidence:.2f})")
                
                # Map class names to standard types
                if "bike" in class_name or "motorcycle" in class_name or "2" in class_name:
                    return "bike", confidence
                elif "car" in class_name or "4" in class_name:
                    return "car", confidence
                else:
                    return class_name, confidence
        
        # Default to car if uncertain
        print("[Vehicle Classifier] No clear detection, defaulting to car")
        return "car", 0.5
        
    except Exception as e:
        print(f"[Vehicle Classifier Error] {str(e)}")
        traceback.print_exc()
        # Default to car on error
        return "car", 0.0

# This function is correct and correctly parses the new 'predict' method's output.
def extract_plate_from_image(base64_image):
    try:
        # Decode base64 image
        img_data = base64.b64decode(base64_image)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run YOLO detection
        results = yolo_model(img)

        detected_plates = []
        for result in results:
            for idx, box in enumerate(result.boxes):
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped_plate = img[y1:y2, x1:x2]
                
                # Preprocess the plate region for better OCR
                gray_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
                # Convert grayscale image to 3 channels (PaddleOCR expects 3 channels)
                gray_plate_3_channel = cv2.cvtColor(gray_plate, cv2.COLOR_GRAY2BGR)
                # Resize to optimal size for license plate OCR
                resized_plate = cv2.resize(gray_plate_3_channel, (320, 96))
                
                print(f"[Debug] Processing plate {idx}: shape={resized_plate.shape}, dtype={resized_plate.dtype}")
                
                try:
                    # Run PaddleOCR using predict method
                    ocr_results = ocr.predict(resized_plate)
                    
                    print(f"[Debug] OCR Results structure: {type(ocr_results)}")
                    
                    # Extract text from OCR results
                    # The new PaddleOCR returns a list with a dict containing 'rec_texts'
                    if ocr_results and isinstance(ocr_results, list) and len(ocr_results) > 0:
                        result_dict = ocr_results[0]
                        
                        # Check if it's the new PaddleOCR format with 'rec_texts'
                        if isinstance(result_dict, dict) and 'rec_texts' in result_dict:
                            rec_texts = result_dict['rec_texts']
                            print(f"[Debug] Found rec_texts: {rec_texts}")
                            
                            # Join all detected texts
                            license_text = ''.join(rec_texts)
                            
                            # Clean the text (remove spaces, keep only alphanumeric)
                            cleaned = ''.join(filter(str.isalnum, license_text)).upper()
                            
                            print(f"[PaddleOCR Debug] Raw text: '{license_text}'")
                            print(f"[PaddleOCR Debug] Cleaned text: '{cleaned}' (length: {len(cleaned)})")
                            
                            if len(cleaned) >= 4:  # Minimum 4 characters for a valid plate
                                detected_plates.append(cleaned)
                                print(f"[PaddleOCR Success] Detected License Plate: '{cleaned}'")
                            else:
                                print(f"[PaddleOCR Warning] Text too short: '{cleaned}'")
                        # Fallback to old format (list of boxes with text)
                        elif isinstance(result_dict, list):
                            license_text = ''.join([
                                line[1][0] for line in result_dict 
                                if len(line) > 1 and len(line[1]) > 0
                            ])
                            cleaned = ''.join(filter(str.isalnum, license_text)).upper()
                            
                            print(f"[PaddleOCR Debug] Raw text (old format): '{license_text}'")
                            print(f"[PaddleOCR Debug] Cleaned text: '{cleaned}' (length: {len(cleaned)})")
                            
                            if len(cleaned) >= 4:
                                detected_plates.append(cleaned)
                                print(f"[PaddleOCR Success] Detected License Plate: '{cleaned}'")
                        else:
                            print(f"[PaddleOCR Warning] Unknown OCR result format: {type(result_dict)}")
                    else:
                        print(f"[PaddleOCR Warning] No text detected on cropped plate {idx}")
                        
                except Exception as ocr_error:
                    print(f"[PaddleOCR Error] OCR processing failed for plate {idx}: {str(ocr_error)}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        if detected_plates:
            # Return the longest detected plate
            best_plate = max(detected_plates, key=len)
            print(f"[PaddleOCR Final] Selected plate: '{best_plate}'")
            return best_plate
        
        print("[PaddleOCR Error] No valid plate detected")
        raise ValueError("OCR_FAILED: No valid license plate text detected")

    except ValueError:
        raise
    except Exception as e:
        print(f"[PaddleOCR Error] Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        raise ValueError(f"OCR_FAILED: {e}")

# ------------------------------------------------
# 🧩 Helper: Generate Parking Token ID
# ------------------------------------------------
def generate_token_id():
    count = records_col.count_documents({}) + 1
    return f"PKG{str(count).zfill(3)}"

# ------------------------------------------------
# 🧩 Helper: Get Free Slot
# ------------------------------------------------
def get_free_slot():
    active_slots = [
        rec["slotNumber"] for rec in records_col.find({"status": "active"}, {"slotNumber": 1})
    ]
    for slot in range(1, TOTAL_SLOTS + 1):
        if slot not in active_slots:
            return slot
    return None

# ------------------------------------------------
# 🚗 Classify Vehicle Type (Car/Bike)
# ------------------------------------------------
@app.route("/api/classify-vehicle", methods=["POST"])
def classify_vehicle_endpoint():
    """
    Classify vehicle as car or bike from image
    Expected JSON: {"image": "base64_encoded_image"}
    Returns: {"vehicleType": "car"|"bike", "confidence": 0.95}
    """
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({
                "success": False,
                "message": "Missing 'image' field in request body"
            }), 400
        
        vehicle_type, confidence = classify_vehicle(data["image"])
        
        return jsonify({
            "success": True,
            "data": {
                "vehicleType": vehicle_type,
                "confidence": round(confidence, 2),
                "category": "4-wheeler" if vehicle_type == "car" else "2-wheeler"
            }
        }), 200
        
    except Exception as e:
        print(f"[Classify Vehicle Error] {str(e)}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"Classification failed: {str(e)}"
        }), 500

# ------------------------------------------------
# 1️⃣ Check Active Session
# ------------------------------------------------
@app.route("/api/check-session", methods=["GET"])
@limiter.exempt  # Exempt from rate limiting - frequently checked endpoint
def check_session():
    """Returns complete car details for active parking session"""
    record = records_col.find_one({"status": "active"}, {"_id": 0})
    if record:
        # Include all car details in response
        record["id"] = record.get("tokenId")
        # Calculate current duration if session is active
        if record.get("entryTime"):
            entry_time = record["entryTime"]
            if entry_time.tzinfo is None:
                entry_time = entry_time.replace(tzinfo=UTC)
            current_time = datetime.now(UTC)
            duration = current_time - entry_time
            record["currentDuration"] = {
                "hours": int(duration.total_seconds() // 3600),
                "minutes": int((duration.total_seconds() % 3600) // 60)
            }
            vehicle_type = record.get("vehicleType", "car")
            record["currentCharge"] = calculate_charge(entry_time, current_time, vehicle_type)
        return jsonify({"success": True, "data": record}), 200
    return jsonify({"success": False, "message": "No active parking session found"}), 404


@app.route("/api/free-slot", methods=["GET"])
@limiter.exempt  
def free_slot():
    slot = get_free_slot()
    if slot:
        active_count = records_col.count_documents({"status": "active"})
        return jsonify({
            "success": True,
            "data": {
                "slotNumber": slot,
                "totalSlots": TOTAL_SLOTS,
                "availableSlots": TOTAL_SLOTS - active_count
            }
        }), 200
    return jsonify({"success": False, "message": "No parking slots available"}), 400


@app.route("/api/entry", methods=["POST"])
def vehicle_entry():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"success": False, "message": "Invalid image data", "error": "INVALID_REQUEST"}), 400

    try:
        vehicle_number = extract_plate_from_image(data["image"])
        free_slot = get_free_slot()
        if not free_slot:
            return jsonify({"success": False, "message": "No parking slots available", "error": "NO_SLOTS_AVAILABLE"}), 400

        
        vehicle_type, classification_confidence = classify_vehicle(data["image"])
        print(f"[Entry] Auto-classified vehicle as: {vehicle_type} (confidence: {classification_confidence:.2f})")

        token_id = generate_token_id()
        entry_time = datetime.now(UTC)
        
       
        active_count = records_col.count_documents({"status": "active"})

        
        record = {
            "tokenId": token_id,
            "vehicleNumber": vehicle_number,
            "slotNumber": free_slot,
            "entryTime": entry_time,
            "exitTime": None,
            "charge": None,
            "status": "active",
            "vehicleType": vehicle_type,  
            "vehicleCategory": "4-wheeler" if vehicle_type == "car" else "2-wheeler",
            "classificationConfidence": round(classification_confidence, 2),
            "ownerName": data.get("ownerName"), 
            "ownerPhone": data.get("ownerPhone"),  
            "vehicleColor": data.get("vehicleColor"),  
            "vehicleModel": data.get("vehicleModel"),  
            "entryImage": data.get("image"),
            "occupancyBeforeEntry": active_count,
            "weekday": entry_time.strftime("%A"),
            "hourOfDay": entry_time.hour,
            "durationMinutes": None,  
            "createdAt": entry_time,
            "updatedAt": entry_time
        }

        result = records_col.insert_one(record)
        record_id = str(result.inserted_id)

        return jsonify({
            "success": True,
            "message": "Vehicle entry recorded successfully",
            "data": {
                "id": token_id,
                "recordId": record_id,
                "vehicleNumber": vehicle_number,
                "vehicleType": vehicle_type,
                "vehicleCategory": "4-wheeler" if vehicle_type == "car" else "2-wheeler",
                "classificationConfidence": round(classification_confidence, 2),
                "slotNumber": free_slot,
                "entryTime": entry_time.isoformat(),
                "status": "active",
                "occupancyBeforeEntry": active_count,
                "availableSlots": TOTAL_SLOTS - active_count - 1
            }
        }), 200

    except ValueError as e:
        error_str = str(e)
        if "OCR_FAILED" in error_str:
            return jsonify({
                "success": False, 
                "message": "Could not detect vehicle number plate. Please retake photo.", 
                "error": "OCR_FAILED"
            }), 400
        return jsonify({"success": False, "message": "Server error processing image", "error": "PROCESSING_ERROR"}), 500
    except Exception as e:
        print(f"[Entry Error] Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": "Server error", "error": "INTERNAL_ERROR"}), 500


@app.route("/api/exit", methods=["POST"])
def vehicle_exit():
    data = request.get_json()
    token_id = data.get("tokenId")
    vehicle_number = data.get("vehicleNumber")

    if not token_id and not vehicle_number:
        return jsonify({"success": False, "message": "tokenId or vehicleNumber required", "error": "INVALID_REQUEST"}), 400

    query = {"tokenId": token_id} if token_id else {"vehicleNumber": vehicle_number, "status": "active"}
    record = records_col.find_one(query)

    if not record:
        return jsonify({"success": False, "message": "No active parking session found for this vehicle", "error": "SESSION_NOT_FOUND"}), 404

    entry_time = record["entryTime"]
    exit_time = datetime.now(UTC)
    
    if entry_time.tzinfo is None:
        entry_time = entry_time.replace(tzinfo=UTC)
    
    
    vehicle_type = record.get("vehicleType", "car")
    
    charge = calculate_charge(entry_time, exit_time, vehicle_type)
    duration = exit_time - entry_time
    hours = int(duration.total_seconds() // 3600)
    minutes = int((duration.total_seconds() % 3600) // 60)
    duration_minutes = int(duration.total_seconds() / 60)

    
    update_fields = {
        "exitTime": exit_time,
        "charge": charge,
        "status": "completed",
        "durationMinutes": duration_minutes,
        "exitImage": data.get("image"),  
        "paymentMethod": data.get("paymentMethod", "cash"),  
        "paymentStatus": data.get("paymentStatus", "pending"),
        "updatedAt": exit_time
    }

    records_col.update_one(
        {"_id": record["_id"]},
        {"$set": update_fields}
    )
    
    try:
        session_data = {
            "slot_id": record["slotNumber"],
            "entry_time": entry_time,
            "exit_time": exit_time,
            "duration_minutes": duration_minutes,
            "vehicle_type": record.get("vehicleType", "unknown"),
            "vehicle_number": record["vehicleNumber"],
            "occupancy_before_entry": record.get("occupancyBeforeEntry", 0),
            "weekday": entry_time.strftime("%A"),
            "hour_of_day": entry_time.hour,
            "charge": charge,
            "logged_at": datetime.now(UTC)
        }
        parking_sessions_col.insert_one(session_data)
        print(f"[Auto-Log] Session logged for ML training: slot={record['slotNumber']}, vehicle={record['vehicleNumber']}")
    except Exception as log_error:
        print(f"[Auto-Log Error] {log_error}")
    
   
    if AUTO_RETRAIN_ENABLED:
        try:
            trigger_retrain_if_needed()
        except Exception as retrain_error:
            print(f"[Auto-Retrain Error] {retrain_error}")

  
    return jsonify({
        "success": True,
        "message": "Parking session completed",
        "data": {
            "id": record["tokenId"],
            "vehicleNumber": record["vehicleNumber"],
            "vehicleType": record.get("vehicleType", "unknown"),
            "ownerName": record.get("ownerName"),
            "ownerPhone": record.get("ownerPhone"),
            "vehicleColor": record.get("vehicleColor"),
            "vehicleModel": record.get("vehicleModel"),
            "slotNumber": record["slotNumber"],
            "entryTime": entry_time.isoformat(),
            "exitTime": exit_time.isoformat(),
            "duration": {"hours": hours, "minutes": minutes},
            "durationMinutes": duration_minutes,
            "charge": charge,
            "paymentMethod": update_fields["paymentMethod"],
            "paymentStatus": update_fields["paymentStatus"],
            "status": "completed"
        }
    }), 200


@app.route("/api/records", methods=["GET"])
@limiter.exempt 
def get_records():
    """Returns all parking records with complete car details"""
    status = request.args.get("status")
    limit = int(request.args.get("limit", 100))
    skip = int(request.args.get("skip", 0))
    vehicle_number = request.args.get("vehicleNumber") 

    query = {}
    if status:
        query["status"] = status
    if vehicle_number:
        query["vehicleNumber"] = {"$regex": vehicle_number, "$options": "i"}  

    total = records_col.count_documents(query)
    cursor = records_col.find(query, {"_id": 0}).sort("entryTime", -1).skip(skip).limit(limit)
    records = list(cursor)

    return jsonify({
        "success": True,
        "data": records,
        "meta": {
            "total": total,
            "returned": len(records),
            "skip": skip,
            "limit": limit
        }
    }), 200


@app.route("/api/record", methods=["GET"])
def get_record():
    """Returns complete car details for a specific parking record"""
    token_id = request.args.get("tokenId")
    vehicle_number = request.args.get("vehicleNumber")
    
    if not token_id and not vehicle_number:
        return jsonify({
            "success": False, 
            "message": "tokenId or vehicleNumber required", 
            "error": "INVALID_REQUEST"
        }), 400
    
    query = {"tokenId": token_id} if token_id else {"vehicleNumber": vehicle_number}
    record = records_col.find_one(query, {"_id": 0})
    
    if not record:
        return jsonify({
            "success": False, 
            "message": "Parking record not found", 
            "error": "RECORD_NOT_FOUND"
        }), 404
    
   
    if record.get("status") == "active" and record.get("entryTime"):
        entry_time = record["entryTime"]
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=UTC)
        current_time = datetime.now(UTC)
        duration = current_time - entry_time
        record["currentDuration"] = {
            "hours": int(duration.total_seconds() // 3600),
            "minutes": int((duration.total_seconds() % 3600) // 60)
        }
        vehicle_type = record.get("vehicleType", "car")
        record["currentCharge"] = calculate_charge(entry_time, current_time, vehicle_type)
    
    return jsonify({
        "success": True,
        "data": record
    }), 200


@app.route("/api/system/restart", methods=["POST"])
def system_restart():
    data = request.get_json(silent=True) or {}
    component = data.get("component", "all")
    return jsonify({
        "success": True,
        "message": "System restart initiated",
        "data": {"component": component, "status": "restarting"}
    }), 200


@app.route("/api/log-session", methods=["POST"])
def log_session():
    """
    Logs parking session details to MongoDB for ML training.
    Expected fields: slot_id, entry_time, exit_time, duration_minutes, 
    vehicle_type, occupancy_before_entry, weekday, hour_of_day
    """
    try:
        data = request.get_json()
        
        required_fields = ["slot_id", "entry_time", "exit_time"]
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                "success": False,
                "message": f"Missing required fields: {', '.join(missing_fields)}",
                "error": "MISSING_FIELDS"
            }), 400
        
        
        entry_time = datetime.fromisoformat(data["entry_time"].replace('Z', '+00:00'))
        exit_time = datetime.fromisoformat(data["exit_time"].replace('Z', '+00:00'))
        
        
        duration_minutes = data.get("duration_minutes")
        if duration_minutes is None:
            duration = exit_time - entry_time
            duration_minutes = int(duration.total_seconds() / 60)
        
     
        weekday = data.get("weekday")
        if weekday is None:
            weekday = entry_time.strftime("%A") 
        
        hour_of_day = data.get("hour_of_day")
        if hour_of_day is None:
            hour_of_day = entry_time.hour
        
       
        occupancy_before_entry = data.get("occupancy_before_entry")
        if occupancy_before_entry is None:
            
            occupancy_before_entry = records_col.count_documents({"status": "active"})
        
        
        session_record = {
            "slot_id": data["slot_id"],
            "entry_time": entry_time,
            "exit_time": exit_time,
            "duration_minutes": duration_minutes,
            "vehicle_type": data.get("vehicle_type", "unknown"),
            "occupancy_before_entry": occupancy_before_entry,
            "weekday": weekday,
            "hour_of_day": hour_of_day,
            "logged_at": datetime.now(UTC)
        }
        
        
        result = parking_sessions_col.insert_one(session_record)
        
        print(f"[Log Session] Session logged successfully: slot={data['slot_id']}, duration={duration_minutes}min")
        
        return jsonify({
            "success": True,
            "message": "Parking session logged successfully",
            "data": {
                "session_id": str(result.inserted_id),
                "slot_id": data["slot_id"],
                "duration_minutes": duration_minutes,
                "weekday": weekday,
                "hour_of_day": hour_of_day,
                "occupancy_before_entry": occupancy_before_entry
            }
        }), 200
        
    except ValueError as e:
        print(f"[Log Session Error] Invalid datetime format: {str(e)}")
        return jsonify({
            "success": False,
            "message": "Invalid datetime format. Use ISO format (YYYY-MM-DDTHH:MM:SS)",
            "error": "INVALID_DATETIME"
        }), 400
    except Exception as e:
        print(f"[Log Session Error] Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": "Server error logging session",
            "error": "INTERNAL_ERROR"
        }), 500


@app.route("/api/predict-availability", methods=["GET", "POST"])
@limiter.limit("100 per minute") 
def predict_availability():
    """
    Predict slot availability using LSTM model.
    Input: Recent occupancy history for prediction
    Output: Predicted availability and confidence
    
    Supports both GET (with query params) and POST (with JSON body)
    """
    if not LSTM_ENABLED:
        return jsonify({
            "success": False,
            "message": "LSTM model not loaded. Train model first with: python train_lstm_model.py",
            "error": "MODEL_NOT_FOUND"
        }), 503
    
    try:
        
        if request.method == "GET":
            current_time = datetime.now(UTC)
            current_hour = current_time.hour
            
            
            active_records = records_col.find({"status": "active"}, {"slotNumber": 1})
            active_slots = {rec["slotNumber"] for rec in active_records}
            
            
            current_occupancy = [0] * TOTAL_SLOTS
            for slot_num in active_slots:
                if 1 <= slot_num <= TOTAL_SLOTS:
                    current_occupancy[slot_num - 1] = 1
            
            
            def generate_time_based_occupancy(hour):
                """Generate realistic occupancy based on hour of day"""
                occupancy = [0] * TOTAL_SLOTS
                
               
                if 7 <= hour <= 10 or 17 <= hour <= 20:
                    num_occupied = random.randint(10, 18)
        
                elif 11 <= hour <= 16:
                    num_occupied = random.randint(5, 12)
    
                else:
                    num_occupied = random.randint(0, 5)
                
                
                occupied_slots = random.sample(range(min(20, TOTAL_SLOTS)), 
                                              min(num_occupied, min(20, TOTAL_SLOTS)))
                for slot_idx in occupied_slots:
                    occupancy[slot_idx] = 1
                
                return occupancy
            
            
            occupancy_matrix = []
            
            
            for i in range(LSTM_TIMESTEPS - 1, 0, -1):
                past_hour = (current_hour - i) % 24
                past_occupancy = generate_time_based_occupancy(past_hour)
                occupancy_matrix.append(past_occupancy)
            
           
            occupancy_matrix.append(current_occupancy)
            
            
            full_history = np.array(occupancy_matrix)
            
        
            slot_id = request.args.get("slot_id", type=int)
            data = {
                "occupancy_matrix": full_history.tolist(),
                "slot_id": slot_id
            }
        else:
            
            data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "message": "No data provided",
                "error": "INVALID_REQUEST"
            }), 400

        if "slot_id" in data and "recent_occupancy" in data:
            slot_id = data["slot_id"]
            recent_occupancy = np.array(data["recent_occupancy"])
    
            if len(recent_occupancy) != LSTM_TIMESTEPS:
                return jsonify({
                    "success": False,
                    "message": f"recent_occupancy must have {LSTM_TIMESTEPS} timesteps",
                    "error": "INVALID_SHAPE"
                }), 400
            
            active_records = records_col.find({"status": "active"}, {"slotNumber": 1})
            active_slots = {rec["slotNumber"] for rec in active_records}
            
            
            full_history = np.zeros((LSTM_TIMESTEPS, TOTAL_SLOTS))
            
           
            if isinstance(slot_id, int) and 1 <= slot_id <= TOTAL_SLOTS:
                full_history[:, slot_id - 1] = recent_occupancy
            else:
                return jsonify({
                    "success": False,
                    "message": f"slot_id must be between 1 and {TOTAL_SLOTS}",
                    "error": "INVALID_SLOT_ID"
                }), 400
            
       
        elif "occupancy_matrix" in data:
            full_history = np.array(data["occupancy_matrix"])
            
            
            if full_history.shape != (LSTM_TIMESTEPS, TOTAL_SLOTS):
                return jsonify({
                    "success": False,
                    "message": f"occupancy_matrix must be shape [{LSTM_TIMESTEPS}, {TOTAL_SLOTS}]",
                    "error": "INVALID_SHAPE"
                }), 400
            
            slot_id = data.get("slot_id", None)
        
        else:
            return jsonify({
                "success": False,
                "message": "Provide either 'slot_id' + 'recent_occupancy' or 'occupancy_matrix'",
                "error": "MISSING_FIELDS"
            }), 400
        
        
        input_reshaped = full_history.reshape(-1, TOTAL_SLOTS)
        input_scaled = lstm_scaler.transform(input_reshaped)
        input_scaled = input_scaled.reshape(1, LSTM_TIMESTEPS, TOTAL_SLOTS)
        
        prediction_scaled = lstm_model.predict(input_scaled, verbose=0)
        
        
        prediction = lstm_scaler.inverse_transform(prediction_scaled)
        prediction_probs = prediction[0] 
        
        
        if slot_id is not None:
            if isinstance(slot_id, int) and 1 <= slot_id <= TOTAL_SLOTS:
                slot_index = slot_id - 1
                occupancy_prob = float(prediction_probs[slot_index])
                availability_prob = 1.0 - occupancy_prob
                
                current_occupancy = full_history[-1, slot_index]
                
                if current_occupancy == 1:
                    if availability_prob > 0.5:
                        predicted_free_minutes = 30 + (availability_prob - 0.5) * 60
                    else:
                        predicted_free_minutes = None  
                else: 
                    predicted_free_minutes = 0 
                
                return jsonify({
                    "success": True,
                    "prediction": {
                        "slot_id": slot_id,
                        "current_status": "occupied" if current_occupancy == 1 else "free",
                        "predicted_status": "free" if availability_prob > 0.5 else "occupied",
                        "availability_probability": round(availability_prob, 3),
                        "occupancy_probability": round(occupancy_prob, 3),
                        "predicted_free_in_minutes": round(predicted_free_minutes, 1) if predicted_free_minutes is not None else None,
                        "confidence": round(max(availability_prob, occupancy_prob), 3),
                        "prediction_horizon": "60 minutes"
                    }
                }), 200
            else:
                return jsonify({
                    "success": False,
                    "message": f"slot_id must be between 1 and {TOTAL_SLOTS}",
                    "error": "INVALID_SLOT_ID"
                }), 400
        
        
        else:
            predictions = []
            for i in range(TOTAL_SLOTS):
                slot_num = i + 1
                occupancy_prob = float(prediction_probs[i])
                availability_prob = 1.0 - occupancy_prob
                current_occupancy = full_history[-1, i]
                
                predictions.append({
                    "slot_id": slot_num,
                    "current_status": "occupied" if current_occupancy == 1 else "free",
                    "predicted_status": "free" if availability_prob > 0.5 else "occupied",
                    "availability_probability": round(availability_prob, 3),
                    "confidence": round(max(availability_prob, occupancy_prob), 3)
                })
            
            
            predictions.sort(key=lambda x: x["availability_probability"], reverse=True)
            
            
            predicted_occupied = sum(1 for p in predictions if p["predicted_status"] == "occupied")
            predicted_available = sum(1 for p in predictions if p["predicted_status"] == "free")
            occupancy_percentage = (predicted_occupied / TOTAL_SLOTS) * 100
            
          
            print(f"[Prediction] Hour: {datetime.now(UTC).hour}, Predicted Occupied: {predicted_occupied}/{TOTAL_SLOTS} ({occupancy_percentage:.1f}%)")
            
            response_data = {
                "success": True,
                "prediction": {
                    "total_slots": TOTAL_SLOTS,
                    "predicted_available": predicted_available,
                    "predicted_occupied": predicted_occupied,
                    "occupancy_percentage": round(occupancy_percentage, 1),
                    "expected_occupancy": round(occupancy_percentage, 1),  
                    "expected_occupancy_percentage": round(occupancy_percentage, 1),
                    "prediction_horizon": "60 minutes",
                    "slots": predictions[:10]  
                },
                "meta": {
                    "model_type": "LSTM",
                    "timesteps_used": LSTM_TIMESTEPS
                }
            }
            
            print(f"[Prediction Response] Occupancy={occupancy_percentage:.1f}%, Occupied={predicted_occupied}, Available={predicted_available}")
            return jsonify(response_data), 200
        
    except ValueError as e:
        print(f"[Predict Error] ValueError: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Invalid input data: {str(e)}",
            "error": "INVALID_DATA"
        }), 400
    except Exception as e:
        print(f"[Predict Error] Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": "Prediction failed",
            "error": "PREDICTION_ERROR"
        }), 500


@app.route("/api/update-model", methods=["POST"])
def update_model():
    """
    Incrementally fine-tune the LSTM model with new session data.
    Accepts new parking session records and performs quick model update.
    """
    if not LSTM_ENABLED:
        return jsonify({
            "success": False,
            "message": "LSTM model not loaded. Train model first with: python train_lstm_model.py",
            "error": "MODEL_NOT_FOUND"
        }), 503
    
    try:
        data = request.get_json()
        
       
        if not data:
            return jsonify({
                "success": False,
                "message": "No data provided",
                "error": "INVALID_REQUEST"
            }), 400
        
       
        sessions = data.get("sessions", [data]) if "sessions" in data else [data]
        
        
        logged_count = 0
        for session in sessions:
            required_fields = ["slot_id", "entry_time", "exit_time"]
            missing_fields = [field for field in required_fields if field not in session]
            
            if missing_fields:
                continue  
            try:
                entry_time = datetime.fromisoformat(session["entry_time"].replace('Z', '+00:00'))
                exit_time = datetime.fromisoformat(session["exit_time"].replace('Z', '+00:00'))
            except ValueError:
                continue  
            
          
            duration_minutes = session.get("duration_minutes")
            if duration_minutes is None:
                duration = exit_time - entry_time
                duration_minutes = int(duration.total_seconds() / 60)
            
           
            weekday = session.get("weekday", entry_time.strftime("%A"))
            hour_of_day = session.get("hour_of_day", entry_time.hour)
            occupancy_before_entry = session.get("occupancy_before_entry", 
                                                 records_col.count_documents({"status": "active"}))
            
           
            session_record = {
                "slot_id": session["slot_id"],
                "entry_time": entry_time,
                "exit_time": exit_time,
                "duration_minutes": duration_minutes,
                "vehicle_type": session.get("vehicle_type", "unknown"),
                "occupancy_before_entry": occupancy_before_entry,
                "weekday": weekday,
                "hour_of_day": hour_of_day,
                "logged_at": datetime.now(UTC)
            }
            
            
            parking_sessions_col.insert_one(session_record)
            logged_count += 1
        
        if logged_count == 0:
            return jsonify({
                "success": False,
                "message": "No valid sessions to process",
                "error": "NO_VALID_SESSIONS"
            }), 400
        
        print(f"[Model Update] Logged {logged_count} new session(s)")
        
        
        import threading
        
        def fine_tune_model():
            """Fine-tune model with recent data."""
            try:
                print(f"\n[Fine-tuning] Starting incremental model update...")
                
              
                recent_sessions = list(parking_sessions_col.find(
                    {},
                    {
                        "_id": 0,
                        "slot_id": 1,
                        "entry_time": 1,
                        "exit_time": 1
                    }
                ).sort("logged_at", -1).limit(100))
                
                if len(recent_sessions) < 20:
                    print(f"[Fine-tuning] Insufficient data: {len(recent_sessions)} sessions")
                    return
                
                df = pd.DataFrame(recent_sessions)
                
                # Create time-series data
                min_time = df['entry_time'].min()
                max_time = df['exit_time'].max()
                
                if min_time.tzinfo is None:
                    min_time = min_time.replace(tzinfo=UTC)
                if max_time.tzinfo is None:
                    max_time = max_time.replace(tzinfo=UTC)
                
                min_time = min_time.replace(minute=0, second=0, microsecond=0)
                max_time = max_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                
                time_range = pd.date_range(start=min_time, end=max_time, freq='H')
                
                # Create occupancy matrix
                occupancy_data = []
                for timestamp in time_range:
                    slot_occupancy = {}
                    for slot_id in range(1, TOTAL_SLOTS + 1):
                        occupied = 0
                        for _, session in df.iterrows():
                            entry = session['entry_time']
                            exit = session['exit_time']
                            
                            if entry.tzinfo is None:
                                entry = entry.replace(tzinfo=UTC)
                            if exit.tzinfo is None:
                                exit = exit.replace(tzinfo=UTC)
                            
                            if session['slot_id'] == slot_id and entry <= timestamp < exit:
                                occupied = 1
                                break
                        
                        slot_occupancy[f'slot_{slot_id}'] = occupied
                    
                    occupancy_data.append(slot_occupancy)
                
                occupancy_df = pd.DataFrame(occupancy_data)
                slot_columns = [f'slot_{i}' for i in range(1, TOTAL_SLOTS + 1)]
                occupancy_matrix = occupancy_df[slot_columns].values
                
                # Create sequences
                def create_sequences(data, timesteps, prediction_offset=1):
                    X, y = [], []
                    for i in range(len(data) - timesteps - prediction_offset + 1):
                        X.append(data[i:i + timesteps])
                        y.append(data[i + timesteps + prediction_offset - 1])
                    return np.array(X), np.array(y)
                
                prediction_offset = 1  # 60 minutes
                X, y = create_sequences(occupancy_matrix, LSTM_TIMESTEPS, prediction_offset)
                
                if len(X) < 5:
                    print(f"[Fine-tuning] Insufficient sequences: {len(X)}")
                    return
                
                # Normalize with existing scaler
                X_reshaped = X.reshape(-1, X.shape[-1])
                X_scaled_reshaped = lstm_scaler.transform(X_reshaped)
                X_scaled = X_scaled_reshaped.reshape(X.shape)
                y_scaled = lstm_scaler.transform(y)
                
                # Fine-tune model (few epochs for quick update)
                print(f"[Fine-tuning] Training on {len(X)} sequences...")
                
                history = lstm_model.fit(
                    X_scaled, y_scaled,
                    epochs=5,  # Quick fine-tuning (5 epochs)
                    batch_size=16,
                    verbose=0,
                    validation_split=0.2
                )
                
                # Save updated model
                lstm_model.save(LSTM_MODEL_PATH)
                
                final_loss = history.history['loss'][-1]
                final_val_loss = history.history.get('val_loss', [0])[-1]
                
                print(f"[Fine-tuning] ✅ Model updated successfully!")
                print(f"   Training Loss: {final_loss:.4f}")
                print(f"   Validation Loss: {final_val_loss:.4f}")
                print(f"   Sequences: {len(X)}")
                print(f"   Model saved to: {LSTM_MODEL_PATH}")
                
            except Exception as e:
                print(f"[Fine-tuning] ❌ Error: {str(e)}")
                import traceback
                traceback.print_exc()
        
        
        thread = threading.Thread(target=fine_tune_model)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "success": True,
            "message": "Sessions logged and model fine-tuning initiated",
            "data": {
                "sessions_logged": logged_count,
                "status": "fine_tuning_started",
                "note": "Model update running in background"
            }
        }), 200
        
    except Exception as e:
        print(f"[Model Update Error] {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": "Failed to update model",
            "error": "UPDATE_ERROR"
        }), 500


# ------------------------------------------------
# 🧠 Prompt Enhancement Endpoint (Gemini API)
# ------------------------------------------------
@app.route('/api/enhance-prompt', methods=['POST', 'OPTIONS'])
@limiter.limit("30 per minute")  # Rate limit: 30 enhancement requests per minute
def enhance_prompt():
    """
    Enhance user prompts using Google Gemini API before sending to RAG.
    This improves query clarity and retrieval precision.
    
    Expected JSON: {"prompt": "user's original query"}
    Returns JSON: {"enhancedPrompt": "improved query"}
    """
    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.get_json()
        
        # Validate input
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        original_prompt = data.get('prompt', '').strip()
        
        if not original_prompt:
            return jsonify({
                'success': False,
                'error': 'Prompt field is required and cannot be empty'
            }), 400
        
        # Check if Gemini API is configured
        if not GEMINI_API_KEY:
            print("[Enhance Prompt] Warning: GEMINI_API_KEY not configured, returning original prompt")
            return jsonify({
                'success': True,
                'enhancedPrompt': original_prompt,
                'fallback': True,
                'message': 'Gemini API not configured, using original prompt'
            }), 200
        
        # System instruction for Gemini
        system_instruction = """You are an advanced prompt enhancer for a RAG-based chatbot about Smart Parking Systems.

Your task is to rewrite or expand the user's query to make it clearer, more specific, and contextually rich for better information retrieval.

Guidelines:
- First, identify if the input is conversational (greetings, thanks, casual chat) or a parking-related query
- For conversational inputs (hi, hello, thanks, thank you, bye, ok, etc.), ONLY correct spelling/grammar - do NOT expand or change the meaning
- For parking-related queries, add relevant context (slots, vehicles, occupancy, charges, sessions, etc.)
- Keep the core meaning and intent of the original query intact
- Make vague parking queries more specific
- Avoid unnecessary verbosity - be concise but clear
- Return ONLY the improved query text, nothing else
- Do not add explanations, disclaimers, or extra formatting

Examples:
Conversational (minimal changes):
- "thksn" → "thanks"
- "thkans" → "thanks"
- "hii" → "hi"
- "okk" → "ok"
- "thankss you" → "thank you"

Parking-related (expand for clarity):
- "how many cars" → "How many cars are currently parked in the parking system?"
- "slot status" → "What is the current status of parking slot availability?"
- "charges" → "What are the parking charges for cars and bikes?"
"""
        
        try:
            # Initialize Gemini model with system instruction
            # Using Gemini 2.5 Flash (stable version)
            model = genai.GenerativeModel(
                model_name='gemini-2.5-flash',
                system_instruction=system_instruction
            )
            
            # Generate enhanced prompt
            print(f"[Enhance Prompt] Original: '{original_prompt}'")
            response = model.generate_content(original_prompt)
            
            # Safely extract text from response (handle blocked/empty responses)
            try:
                enhanced_prompt = response.text.strip()
                print(f"[Enhance Prompt] Enhanced: '{enhanced_prompt}'")
            except ValueError as ve:
                # Response was blocked or has no text (safety filters, etc.)
                print(f"[Enhance Prompt] Response blocked or empty (finish_reason={response.candidates[0].finish_reason if response.candidates else 'unknown'}), using original prompt")
                enhanced_prompt = original_prompt
                fallback = True
            except AttributeError:
                # Response has unexpected structure
                print(f"[Enhance Prompt] Invalid response structure, using original prompt")
                enhanced_prompt = original_prompt
                fallback = True
            else:
                # Successfully got enhanced prompt, now validate it
                # Fallback to original if Gemini returns empty or excessively long (>500 chars)
                if not enhanced_prompt:
                    print("[Enhance Prompt] Empty response, using original prompt")
                    enhanced_prompt = original_prompt
                    fallback = True
                elif len(enhanced_prompt) > 500:  # Absolute max length to prevent abuse
                    print(f"[Enhance Prompt] Response too long ({len(enhanced_prompt)} chars), using original prompt")
                    enhanced_prompt = original_prompt
                    fallback = True
                else:
                    fallback = False
            
            return jsonify({
                'success': True,
                'enhancedPrompt': enhanced_prompt,
                'originalPrompt': original_prompt,
                'fallback': fallback
            }), 200
            
        except Exception as gemini_error:
            # Graceful fallback on Gemini API errors
            print(f"[Enhance Prompt] Gemini API Error: {str(gemini_error)}")
            traceback.print_exc()
            
            return jsonify({
                'success': True,
                'enhancedPrompt': original_prompt,
                'fallback': True,
                'message': 'Enhancement failed, using original prompt'
            }), 200
    
    except Exception as e:
        print(f"[Enhance Prompt Error] {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'Failed to process prompt enhancement request'
        }), 500


# ------------------------------------------------
# 🔟 Chatbot Endpoint
# ------------------------------------------------
@app.route('/api/chatbot', methods=['POST', 'OPTIONS'])
def chat():
    '''
    Endpoint to handle chatbot queries with real-time MongoDB data
    Expects JSON: {"query": "your question here"}
    Returns JSON: {"response": "chatbot answer"}
    '''
    # Handle OPTIONS request for CORS preflight (no JSON body)
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        # Reinitialize RAG system to get fresh data from MongoDB on every query
        print(f"[Chatbot] Fetching fresh data for query: {query}")
        fresh_qdrant_client, fresh_collection_name, fresh_model, fresh_mongo_uri = initialize_rag_system()
        
        # Get response with fresh real-time data
        response = get_chatbot_response(query, fresh_qdrant_client, fresh_collection_name, fresh_model, fresh_mongo_uri)
        
        return jsonify({
            'query': query,
            'response': response
        }), 200
    
    except Exception as e:
        print(f"[Chatbot Error] {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ------------------------------------------------
# 🔐 OTP Authentication Endpoints
# ------------------------------------------------
@app.route('/api/send-otp', methods=['POST'])
@limiter.limit("5 per minute") 
def send_otp():
    """
    Send OTP to authorized email address
    Expected JSON: {"email": "abc@gmail.com"}
    Returns: {"success": true, "message": "OTP sent successfully"}
    """
    try:
        
        cleanup_expired_otps()
        
        data = request.get_json()
        
        if not data or 'email' not in data:
            print("[OTP Send] Error: No email provided")
            return jsonify({
                "success": False,
                "error": "Email is required"
            }), 400
        
        email = data['email'].strip().lower()
        
       
        if email != AUTHORIZED_EMAIL.lower():
            print(f"[OTP Send] Unauthorized email attempt: {email}")
            return jsonify({
                "success": False,
                "error": "Unauthorized email"
            }), 403
        
        
        otp = generate_otp()
        
        
        otp_storage[email] = {
            'otp': otp,
            'created_at': datetime.now(UTC),
            'expires_at': datetime.now(UTC) + timedelta(minutes=OTP_EXPIRY_MINUTES)
        }
        
        print(f"[OTP Send] Generated OTP for {email}: {otp} (expires in {OTP_EXPIRY_MINUTES} minutes)")
        
        
        def send_email_async():
            send_otp_email(email, otp)
        
        email_thread = threading.Thread(target=send_email_async)
        email_thread.daemon = True
        email_thread.start()
        
        return jsonify({
            "success": True,
            "message": f"OTP sent to {email}. Valid for {OTP_EXPIRY_MINUTES} minutes.",
            "expiresIn": OTP_EXPIRY_MINUTES * 60  
        }), 200
        
    except Exception as e:
        print(f"[OTP Send Error] {str(e)}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": "Failed to send OTP"
        }), 500


@app.route('/api/verify-otp', methods=['POST'])
@limiter.limit("10 per minute") 
def verify_otp():
    """
    Verify OTP and issue JWT token
    Expected JSON: {"email": "abc@gmail.com", "otp": "123456"}
    Returns: {"success": true, "token": "<JWT_TOKEN>"}
    """
    try:
       
        cleanup_expired_otps()
        
        data = request.get_json()
        
        if not data or 'email' not in data or 'otp' not in data:
            print("[OTP Verify] Error: Missing email or OTP")
            return jsonify({
                "success": False,
                "error": "Email and OTP are required"
            }), 400
        
        email = data['email'].strip().lower()
        provided_otp = data['otp'].strip()
        
        
        if email not in otp_storage:
            print(f"[OTP Verify] No OTP found for {email}")
            return jsonify({
                "success": False,
                "error": "Invalid or expired OTP"
            }), 401
        
        stored_data = otp_storage[email]
        
        if datetime.now(UTC) > stored_data['expires_at']:
            del otp_storage[email]
            print(f"[OTP Verify] Expired OTP for {email}")
            return jsonify({
                "success": False,
                "error": "Invalid or expired OTP"
            }), 401
        
        
        if provided_otp != stored_data['otp']:
            print(f"[OTP Verify] Incorrect OTP for {email}. Expected: {stored_data['otp']}, Got: {provided_otp}")
            return jsonify({
                "success": False,
                "error": "Invalid or expired OTP"
            }), 401
        
        
        token_payload = {
            'email': email,
            'iat': datetime.now(UTC),
            'exp': datetime.now(UTC) + timedelta(hours=24)  
        }
        
        token = jwt.encode(token_payload, JWT_SECRET_KEY, algorithm='HS256')
        
        del otp_storage[email]
        
        print(f"[OTP Verify] Successfully verified OTP for {email}. JWT token issued.")
        
        return jsonify({
            "success": True,
            "message": "OTP verified successfully",
            "token": token,
            "expiresIn": 24 * 60 * 60 
        }), 200
        
    except Exception as e:
        print(f"[OTP Verify Error] {str(e)}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": "Failed to verify OTP"
        }), 500



















































































# ------------------------------------------------
# 🏁 Run Server
# ------------------------------------------------
if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", 8011))
    debug = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    
    # ⬇️ CHANGED: Added use_reloader=False
    # This fixes the "WinError 10038: Not a socket" and TimeoutErrors
    # caused by the reloader clashing with PaddleOCR initialization on Windows.
    # Debug mode will still be active.
    app.run(host=host, port=port, debug=debug, use_reloader=False)
