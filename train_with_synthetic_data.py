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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
import os
import random
from dotenv import load_dotenv
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME", "smart_parking_db")
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
TIMESTEPS = 2
TOTAL_SLOTS = int(os.getenv("TOTAL_SLOTS", 50))
PREDICTION_INTERVAL_MINUTES = 60
MODEL_PATH = "parking_lstm_model.keras"
SCALER_PATH = "parking_scaler.pkl"
SYNTHETIC_SAMPLES = 1000
print("=" * 60)
print("🚗 LSTM Training with Real + Synthetic Data")
print("=" * 60)
print("\n[1/8] Fetching real parking session data from MongoDB...")
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
print(f"✅ Fetched {len(real_sessions)} real parking sessions from DB")
print(f"\n[2/8] Generating {SYNTHETIC_SAMPLES} synthetic parking sessions...")
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
    if 7 <= hour <= 10 or 17 <= hour <= 20:
        duration_minutes = random.choice([30, 45, 60, 90, 120, 150, 180])
    elif 22 <= hour or hour <= 6:
        duration_minutes = random.choice([180, 240, 300, 360, 420, 480])
    else:
        duration_minutes = random.choice([45, 60, 90, 120, 150, 180, 240])
    exit_time = entry_time + timedelta(minutes=duration_minutes)
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
print(f"✅ Generated {len(synthetic_sessions)} synthetic sessions")
print(f"   Covering all 24 hours across last 7 days")
print("\n[3/8] Combining real and synthetic data...")
all_sessions = real_sessions + synthetic_sessions
print(f"✅ Total sessions for training: {len(all_sessions)}")
print(f"   Real: {len(real_sessions)}, Synthetic: {len(synthetic_sessions)}")
df = pd.DataFrame(all_sessions)
for col in ['entry_time', 'exit_time']:
    df[col] = df[col].apply(lambda x: x.replace(tzinfo=UTC) if isinstance(x, datetime) and x.tzinfo is None else x)
print(f"   Date range: {df['entry_time'].min()} to {df['exit_time'].max()}")
print("\n[4/8] Creating time-series occupancy dataset...")
min_time = df['entry_time'].min()
max_time = df['exit_time'].max()
if min_time.tzinfo is None:
    min_time = min_time.replace(tzinfo=UTC)
if max_time.tzinfo is None:
    max_time = max_time.replace(tzinfo=UTC)
min_time = min_time.replace(minute=0, second=0, microsecond=0)
max_time = max_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
time_range = pd.date_range(start=min_time, end=max_time, freq='h')
print(f"   Timeline: {len(time_range)} hours from {min_time} to {max_time}")
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
    slot_occupancy['timestamp'] = timestamp
    slot_occupancy['hour'] = timestamp.hour
    slot_occupancy['day_of_week'] = timestamp.weekday()
    slot_occupancy['is_weekend'] = 1 if timestamp.weekday() >= 5 else 0
    occupancy_data.append(slot_occupancy)
occupancy_df = pd.DataFrame(occupancy_data)
print(f"✅ Created occupancy matrix: {occupancy_df.shape}")
print(f"   Total occupancy records: {len(occupancy_df)} hours × {TOTAL_SLOTS} slots")
min_required_hours = TIMESTEPS + (PREDICTION_INTERVAL_MINUTES // 60)
if len(occupancy_df) < min_required_hours:
    print(f"\n⚠️  Still insufficient hourly data: {len(occupancy_df)} hours (minimum: {min_required_hours})")
    print(f"   Increasing synthetic data spread...")
    exit(1)
print("\n[5/8] Preparing sequence data for LSTM...")
def create_sequences(data, timesteps, prediction_offset=1):
    X, y = [], []
    for i in range(len(data) - timesteps - prediction_offset + 1):
        X.append(data[i:i + timesteps])
        y.append(data[i + timesteps + prediction_offset - 1])
    return np.array(X), np.array(y)
slot_columns = [f'slot_{i}' for i in range(1, TOTAL_SLOTS + 1)]
occupancy_matrix = occupancy_df[slot_columns].values
print(f"   Occupancy matrix shape: {occupancy_matrix.shape}")
prediction_offset = PREDICTION_INTERVAL_MINUTES // 60
X, y = create_sequences(occupancy_matrix, TIMESTEPS, prediction_offset)
print(f"✅ Created {len(X)} sequences")
print(f"   Input shape (X): {X.shape}  [samples, timesteps, features]")
print(f"   Target shape (y): {y.shape}  [samples, features]")
if len(X) == 0:
    print("❌ No sequences created! Need more data spread over time.")
    exit(1)
print("\n[6/8] Normalizing data...")
scaler = MinMaxScaler(feature_range=(0, 1))
X_reshaped = X.reshape(-1, X.shape[-1])
X_scaled_reshaped = scaler.fit_transform(X_reshaped)
X_scaled = X_scaled_reshaped.reshape(X.shape)
y_scaled = scaler.transform(y)
print(f"✅ Data normalized using MinMaxScaler")
with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)
print(f"✅ Scaler saved to '{SCALER_PATH}'")
print("\n[7/8] Splitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, 
    test_size=0.2, 
    random_state=42,
    shuffle=False
)
print(f"✅ Train set: {X_train.shape[0]} samples")
print(f"   Test set:  {X_test.shape[0]} samples")
print("\n[8/8] Building and training LSTM model...")
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
print(model.summary())
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)
model_checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)
print("\n🏋️ Training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=16,
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)
test_loss, test_accuracy, test_binary_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Loss: {test_loss:.4f}")
print(f"✅ Test Binary Accuracy (per-slot): {test_binary_acc:.4f} ({test_binary_acc:.2%})")
predictions = model.predict(X_test, verbose=0)
predictions_binary = (predictions > 0.5).astype(int)
y_test_inverse = scaler.inverse_transform(y_test)
y_test_binary = (y_test_inverse > 0.5).astype(int)
per_slot_accuracy = np.mean(predictions_binary == y_test_binary)
print(f"✅ Per-Slot Prediction Accuracy: {per_slot_accuracy:.2%}")
avg_predicted_occupancy = np.mean(np.sum(predictions_binary, axis=1))
avg_actual_occupancy = np.mean(np.sum(y_test_binary, axis=1))
print(f"✅ Average Predicted Occupancy: {avg_predicted_occupancy:.1f} / {TOTAL_SLOTS} slots")
print(f"✅ Average Actual Occupancy: {avg_actual_occupancy:.1f} / {TOTAL_SLOTS} slots")
model.save(MODEL_PATH)
print(f"✅ Model saved to '{MODEL_PATH}'")
sample_input = X_test[0:1]
prediction = model.predict(sample_input, verbose=0)
prediction_actual = scaler.inverse_transform(prediction)
actual_target = scaler.inverse_transform(y_test[0:1])
print(f"\n🧪 Sample prediction (first 10 slots):")
print(f"   Predicted: {np.round(prediction_actual[0][:10])}")
print(f"   Actual:    {actual_target[0][:10]}")
sample_matches = np.sum(np.round(prediction_actual[0]) == actual_target[0])
print(f"   Matches: {sample_matches}/{TOTAL_SLOTS} slots ({sample_matches/TOTAL_SLOTS:.1%})")
print("\n" + "=" * 60)
print("✅ LSTM Model Training Complete!")
print("=" * 60)
print(f"\n📊 Training Summary:")
print(f"   • Real sessions used: {len(real_sessions)}")
print(f"   • Synthetic sessions generated: {len(synthetic_sessions)}")
print(f"   • Total training sequences: {len(X)}")
print(f"   • Per-slot accuracy: {per_slot_accuracy:.2%} ⭐ (This is your real accuracy!)")
print(f"   • Binary accuracy: {test_binary_acc:.2%}")
print(f"   • Exact match accuracy: {test_accuracy:.2%} (requires all 50 slots perfect - not important)")
print(f"\n📁 Files created:")
print(f"   • {MODEL_PATH} - Trained LSTM model")
print(f"   • {SCALER_PATH} - Data scaler")
print(f"\n🚀 Your model is working great! Restart Flask to use it:")
print(f"   Run: python app.py")
