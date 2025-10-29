"""
Automatic LSTM Model Retraining System
Retrains the model after every 5 vehicle exits to keep predictions accurate.
This is called from the Flask app's exit endpoint.
"""

import numpy as np
import pandas as pd
from pymongo import MongoClient
from datetime import datetime, timedelta, UTC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import os
from dotenv import load_dotenv
import threading

# Load environment variables
load_dotenv()

# ------------------------------------------------
# 🗄️ MongoDB Configuration
# ------------------------------------------------
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME", "smart_parking_db")

# ------------------------------------------------
# 🔧 Model Configuration
# ------------------------------------------------
TIMESTEPS = 2  # Must match train_with_synthetic_data.py
TOTAL_SLOTS = int(os.getenv("TOTAL_SLOTS", 50))
PREDICTION_INTERVAL_MINUTES = 60
MODEL_PATH = "parking_lstm_model.keras"  # Changed to .keras format
SCALER_PATH = "parking_scaler.pkl"
MIN_SESSIONS_FOR_TRAINING = 5  # Minimum sessions required to train (lowered from 50)
SYNTHETIC_SAMPLES = 1000  # Base synthetic data to always include

# Track exit count
exit_counter_file = "exit_counter.txt"

def get_exit_count():
    """Get current exit count from file."""
    try:
        if os.path.exists(exit_counter_file):
            with open(exit_counter_file, 'r') as f:
                return int(f.read().strip())
        return 0
    except:
        return 0

def increment_exit_count():
    """Increment exit count and return new count."""
    count = get_exit_count() + 1
    with open(exit_counter_file, 'w') as f:
        f.write(str(count))
    return count

def reset_exit_count():
    """Reset exit count to 0."""
    with open(exit_counter_file, 'w') as f:
        f.write('0')

def should_retrain():
    """Check if model should be retrained (every 5 exits)."""
    count = get_exit_count()
    return count >= 5

def create_sequences(data, timesteps, prediction_offset=1):
    """Create sequences for LSTM training."""
    X, y = [], []
    for i in range(len(data) - timesteps - prediction_offset + 1):
        X.append(data[i:i + timesteps])
        y.append(data[i + timesteps + prediction_offset - 1])
    return np.array(X), np.array(y)

def retrain_model():
    """
    Retrain the LSTM model with latest data from MongoDB.
    Uses 1000 synthetic samples + all real data that has accumulated.
    This runs in a background thread to avoid blocking the API.
    """
    try:
        print("\n" + "=" * 70)
        print("🔄 AUTO-RETRAIN: LSTM Model Retraining Started")
        print("=" * 70)
        print(f"⏰ Time: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"📊 Trigger: Every 5th vehicle exit")
        print("=" * 70)
        
        # Connect to MongoDB
        client = MongoClient(
            MONGODB_URI,
            tls=True,
            tlsAllowInvalidCertificates=True,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            socketTimeoutMS=5000
        )
        db = client[DB_NAME]
        parking_sessions_col = db["parking_sessions"]
        
        # Fetch REAL parking sessions from database
        print("\n[Step 1/8] Fetching REAL parking sessions from MongoDB...")
        real_sessions = list(parking_sessions_col.find({}, {
            "_id": 0,
            "slot_id": 1,
            "entry_time": 1,
            "exit_time": 1,
            "duration_minutes": 1,
            "occupancy_before_entry": 1,
            "weekday": 1,
            "hour_of_day": 1
        }))
        
        print(f"✅ Fetched {len(real_sessions)} REAL sessions from database")
        
        # Generate SYNTHETIC data (always 1000 samples as base)
        print(f"\n[Step 2/8] Generating {SYNTHETIC_SAMPLES} SYNTHETIC parking sessions...")
        
        import random
        synthetic_sessions = []
        base_time = datetime.now(UTC) - timedelta(days=7)
        
        for i in range(SYNTHETIC_SAMPLES):
            slot_id = random.randint(1, min(20, TOTAL_SLOTS))
            hour = random.randint(0, 23)
            day_offset = random.randint(0, 6)
            
            entry_time = base_time + timedelta(
                days=day_offset,
                hours=hour,
                minutes=random.randint(0, 59)
            )
            
            # Realistic duration based on time of day
            if 7 <= hour <= 10 or 17 <= hour <= 20:
                duration_minutes = random.choice([30, 45, 60, 90, 120, 150, 180])
            elif 22 <= hour or hour <= 6:
                duration_minutes = random.choice([180, 240, 300, 360, 420, 480])
            else:
                duration_minutes = random.choice([45, 60, 90, 120, 150, 180, 240])
            
            exit_time = entry_time + timedelta(minutes=duration_minutes)
            
            # Occupancy varies by time of day
            if 7 <= hour <= 10 or 17 <= hour <= 20:
                occupancy = random.randint(10, 18)
            elif 11 <= hour <= 16:
                occupancy = random.randint(5, 12)
            else:
                occupancy = random.randint(0, 5)
            
            synthetic_session = {
                "slot_id": slot_id,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "duration_minutes": duration_minutes,
                "occupancy_before_entry": occupancy,
                "weekday": entry_time.strftime("%A"),
                "hour_of_day": entry_time.hour
            }
            synthetic_sessions.append(synthetic_session)
        
        print(f"✅ Generated {len(synthetic_sessions)} SYNTHETIC sessions")
        
        # Combine real + synthetic
        print(f"\n[Step 3/8] Combining REAL + SYNTHETIC data...")
        all_sessions = real_sessions + synthetic_sessions
        total_sessions = len(all_sessions)
        print(f"✅ Total training data: {total_sessions} sessions")
        print(f"   → REAL sessions: {len(real_sessions)}")
        print(f"   → SYNTHETIC sessions: {len(synthetic_sessions)}")
        print(f"   📈 Progress: 1000 → {1000 + len(real_sessions)} (+{len(real_sessions)} real)")
        
        if total_sessions < MIN_SESSIONS_FOR_TRAINING:
            print(f"\n⚠️  Insufficient total data: {total_sessions} sessions (minimum: {MIN_SESSIONS_FOR_TRAINING})")
            print("   Skipping retraining. Collect more data.")
            return False
        
        # Convert to DataFrame
        print(f"\n[Step 4/8] Creating time-series occupancy dataset...")
        df = pd.DataFrame(all_sessions)
        
        # Ensure all datetimes are timezone-aware BEFORE calling min/max
        for col in ['entry_time', 'exit_time']:
            df[col] = pd.to_datetime(df[col])
            # Make timezone-aware if not already
            if df[col].dt.tz is None:
                df[col] = df[col].dt.tz_localize(UTC)
            else:
                df[col] = df[col].dt.tz_convert(UTC)
        
        # Create time-series occupancy data
        min_time = df['entry_time'].min()
        max_time = df['exit_time'].max()
        
        min_time = min_time.replace(minute=0, second=0, microsecond=0)
        max_time = max_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        
        # Create hourly timeline
        time_range = pd.date_range(start=min_time, end=max_time, freq='H')
        print(f"✅ Timeline: {len(time_range)} hours")
        
        # Create occupancy matrix
        occupancy_data = []
        for timestamp in time_range:
            slot_occupancy = {}
            for slot_id in range(1, TOTAL_SLOTS + 1):
                occupied = 0
                for _, session in df.iterrows():
                    entry = session['entry_time']
                    exit = session['exit_time']
                    
                    # Timestamps are already timezone-aware from DataFrame conversion above
                    if session['slot_id'] == slot_id and entry <= timestamp < exit:
                        occupied = 1
                        break
                
                slot_occupancy[f'slot_{slot_id}'] = occupied
            
            slot_occupancy['timestamp'] = timestamp
            occupancy_data.append(slot_occupancy)
        
        occupancy_df = pd.DataFrame(occupancy_data)
        
        # Extract slot columns
        slot_columns = [f'slot_{i}' for i in range(1, TOTAL_SLOTS + 1)]
        occupancy_matrix = occupancy_df[slot_columns].values
        
        # Create sequences
        prediction_offset = PREDICTION_INTERVAL_MINUTES // 60
        X, y = create_sequences(occupancy_matrix, TIMESTEPS, prediction_offset)
        
        if len(X) < 20:
            print(f"⚠️  Insufficient sequences: {len(X)} (minimum: 20)")
            print("   Skipping retraining. Collect more data.")
            return False
        
        print(f"✅ Created {len(X)} sequences")
        
        # Normalize data
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled_reshaped = scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled_reshaped.reshape(X.shape)
        y_scaled = scaler.transform(y)
        
        # Save scaler
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Split data (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, 
            test_size=0.2, 
            random_state=42,
            shuffle=False
        )
        
        print(f"✅ Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
        
        # Build model
        model = Sequential([
            LSTM(128, activation='tanh', return_sequences=True, 
                 input_shape=(TIMESTEPS, TOTAL_SLOTS)),
            Dropout(0.2),
            LSTM(64, activation='tanh', return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(TOTAL_SLOTS, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'binary_accuracy']
        )
        
        # Train with reduced epochs for faster retraining
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=0
        )
        
        print("🏋️  Training model...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=30,  # Reduced from 50 for faster retraining
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0  # Silent training
        )
        
        # Evaluate
        test_loss, test_accuracy, test_binary_acc = model.evaluate(X_test, y_test, verbose=0)
        
        # Save model
        model.save(MODEL_PATH)
        
        print(f"\n✅ AUTO-RETRAIN COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"📊 Training Results:")
        print(f"   • Total training data: {total_sessions} sessions")
        print(f"     → Real sessions: {len(real_sessions)}")
        print(f"     → Synthetic sessions: {len(synthetic_sessions)}")
        print(f"   • Sequences created: {len(X)}")
        print(f"   • Test Loss: {test_loss:.4f}")
        print(f"   • Test Binary Accuracy: {test_binary_acc:.4f} ({test_binary_acc*100:.2f}%)")
        print(f"   • Model saved: {MODEL_PATH}")
        print(f"⏰ Completed at: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print("=" * 70)
        print("🎯 Next retrain will occur after 5 more vehicle exits")
        print("=" * 70 + "\n")
        
        # Reset counter after successful retraining
        reset_exit_count()
        print("✅ Exit counter reset to 0")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Model retraining failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def trigger_retrain_if_needed():
    """
    Check if retraining is needed and trigger it.
    Called from the Flask app after each vehicle exit.
    """
    count = increment_exit_count()
    
    print(f"\n{'='*50}")
    print(f"📊 AUTO-RETRAIN TRACKER")
    print(f"{'='*50}")
    print(f"🚗 Exit count: {count}/5")
    print(f"{'='*50}")
    
    if should_retrain():
        print(f"\n{'🔔'*25}")
        print("🚨 RETRAIN THRESHOLD REACHED!")
        print(f"{'🔔'*25}")
        print("✅ Starting background model retraining...")
        print("⏳ This will take 1-3 minutes")
        print("📈 Model will improve with new real data")
        print(f"{'='*50}\n")
        
        # Run retraining in background thread to avoid blocking the API
        thread = threading.Thread(target=retrain_model)
        thread.daemon = True
        thread.start()
        return True
    else:
        remaining = 5 - count
        print(f"⏳ {remaining} more exit(s) until next retrain")
        print(f"{'='*50}\n")
    return False

# ------------------------------------------------
# 🧪 Manual Trigger (for testing)
# ------------------------------------------------
if __name__ == "__main__":
    print("🔧 Manual LSTM Model Retraining")
    print("This will retrain the model immediately.\n")
    
    success = retrain_model()
    
    if success:
        print("\n✅ Retraining completed successfully!")
    else:
        print("\n❌ Retraining failed or skipped.")
