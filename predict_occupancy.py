"""
LSTM Model Prediction Script for Parking Slot Occupancy
Uses trained model to predict next 60-minute occupancy for parking slots.
"""

import numpy as np
import pickle
from tensorflow import keras
from datetime import datetime, UTC
import os

# ------------------------------------------------
# 🔧 Configuration
# ------------------------------------------------
MODEL_PATH = "parking_lstm_model.h5"
SCALER_PATH = "parking_scaler.pkl"
TIMESTEPS = 10  # Must match training configuration
TOTAL_SLOTS = int(os.getenv("TOTAL_SLOTS", 50))

# ------------------------------------------------
# 📊 Load Model and Scaler
# ------------------------------------------------
def load_model_and_scaler():
    """Load trained LSTM model and scaler."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")
    
    model = keras.models.load_model(MODEL_PATH)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler

# ------------------------------------------------
# 🔮 Predict Next Hour Occupancy
# ------------------------------------------------
def predict_next_hour_occupancy(recent_occupancy_history):
    """
    Predict occupancy for the next hour based on recent history.
    
    Args:
        recent_occupancy_history: numpy array of shape [timesteps, num_slots]
                                  Last TIMESTEPS hours of occupancy data
                                  Each row is occupancy status for all slots (0 or 1)
    
    Returns:
        predicted_occupancy: numpy array of shape [num_slots]
                            Predicted occupancy for each slot (0 or 1)
    """
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    
    # Validate input shape
    if recent_occupancy_history.shape != (TIMESTEPS, TOTAL_SLOTS):
        raise ValueError(
            f"Input shape must be ({TIMESTEPS}, {TOTAL_SLOTS}), "
            f"got {recent_occupancy_history.shape}"
        )
    
    # Normalize input
    input_reshaped = recent_occupancy_history.reshape(-1, TOTAL_SLOTS)
    input_scaled = scaler.transform(input_reshaped)
    input_scaled = input_scaled.reshape(1, TIMESTEPS, TOTAL_SLOTS)
    
    # Predict
    prediction_scaled = model.predict(input_scaled, verbose=0)
    
    # Inverse transform
    prediction = scaler.inverse_transform(prediction_scaled)
    
    # Round to binary (0 or 1)
    prediction_binary = np.round(prediction[0]).astype(int)
    
    return prediction_binary

# ------------------------------------------------
# 🧪 Example Usage
# ------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("🔮 LSTM Parking Occupancy Prediction")
    print("=" * 60)
    
    # Example: Create dummy recent history (last 10 hours of occupancy)
    # In real usage, this would come from your database
    print("\n📊 Creating sample input data...")
    
    # Simulate recent occupancy history (10 timesteps × 50 slots)
    # Random binary occupancy for demonstration
    np.random.seed(42)
    sample_history = np.random.randint(0, 2, size=(TIMESTEPS, TOTAL_SLOTS))
    
    print(f"   Input shape: {sample_history.shape}")
    print(f"   Last hour occupancy (first 10 slots): {sample_history[-1][:10]}")
    
    # Make prediction
    print("\n🔮 Predicting next hour occupancy...")
    try:
        predicted_occupancy = predict_next_hour_occupancy(sample_history)
        
        print(f"✅ Prediction complete!")
        print(f"   Predicted occupancy (first 10 slots): {predicted_occupancy[:10]}")
        
        # Calculate statistics
        total_occupied = np.sum(predicted_occupancy)
        occupancy_rate = (total_occupied / TOTAL_SLOTS) * 100
        
        print(f"\n📊 Prediction Summary:")
        print(f"   Total slots occupied: {total_occupied}/{TOTAL_SLOTS}")
        print(f"   Occupancy rate: {occupancy_rate:.1f}%")
        print(f"   Available slots: {TOTAL_SLOTS - total_occupied}")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("   Please train the model first by running: python train_lstm_model.py")
    except Exception as e:
        print(f"\n❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
