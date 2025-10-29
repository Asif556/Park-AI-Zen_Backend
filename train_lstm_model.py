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
MODEL_PATH = "parking_lstm_model.h5"
SCALER_PATH = "parking_scaler.pkl"
print("=" * 60)
print("🚗 LSTM Parking Occupancy Prediction Model Training")
print("=" * 60)
print("\n[1/7] Fetching parking session data from MongoDB...")
sessions = list(parking_sessions_col.find({}, {
    "_id": 0,
    "slot_id": 1,
    "entry_time": 1,
    "exit_time": 1,
    "duration_minutes": 1,
    "occupancy_before_entry": 1,
    "weekday": 1,
    "hour_of_day": 1
}))
MINIMUM_SESSIONS = 5
if len(sessions) == 0:
    print("❌ No training data found in 'parking_sessions' collection!")
    print("   Please log some parking sessions using /api/log-session endpoint first.")
    exit(1)
if len(sessions) < MINIMUM_SESSIONS:
    print(f"⚠️  Insufficient data: {len(sessions)} sessions (minimum: {MINIMUM_SESSIONS})")
    print(f"   Please collect at least {MINIMUM_SESSIONS - len(sessions)} more parking sessions.")
    exit(1)
print(f"✅ Fetched {len(sessions)} parking sessions")
df = pd.DataFrame(sessions)
print(f"   Columns: {list(df.columns)}")
print(f"   Date range: {df['entry_time'].min()} to {df['exit_time'].max()}")
print("\n[2/7] Creating time-series occupancy dataset...")
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
    print(f"\n⚠️  Insufficient hourly data: {len(occupancy_df)} hours (minimum: {min_required_hours})")
    print(f"   Need sessions spanning at least {min_required_hours} hours.")
    print(f"   Current span: {len(occupancy_df)} hours")
    print(f"   Tip: Log sessions with wider time ranges (different entry/exit times)")
    exit(1)
print("\n[3/7] Preparing sequence data for LSTM...")
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
print("\n[4/7] Normalizing data...")
scaler = MinMaxScaler(feature_range=(0, 1))
X_reshaped = X.reshape(-1, X.shape[-1])
X_scaled_reshaped = scaler.fit_transform(X_reshaped)
X_scaled = X_scaled_reshaped.reshape(X.shape)
y_scaled = scaler.transform(y)
print(f"✅ Data normalized using MinMaxScaler")
print(f"   Scaler range: {scaler.feature_range}")
with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)
print(f"✅ Scaler saved to '{SCALER_PATH}'")
print("\n[5/7] Splitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, 
    test_size=0.2, 
    random_state=42,
    shuffle=False
)
print(f"✅ Train set: {X_train.shape[0]} samples")
print(f"   Test set:  {X_test.shape[0]} samples")
print("\n[6/7] Building LSTM model...")
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
print("\n🏋️ Training model... (this may take a few minutes)")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)
print("\n[7/7] Evaluating and saving model...")
test_loss, test_accuracy, test_binary_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Test Loss: {test_loss:.4f}")
print(f"✅ Test Accuracy: {test_accuracy:.4f}")
print(f"✅ Test Binary Accuracy: {test_binary_acc:.4f}")
model.save(MODEL_PATH)
print(f"✅ Model saved to '{MODEL_PATH}'")
print("\n🧪 Testing prediction on a sample...")
sample_input = X_test[0:1]
prediction = model.predict(sample_input, verbose=0)
print(f"   Input shape: {sample_input.shape}")
print(f"   Prediction shape: {prediction.shape}")
print(f"   Sample prediction (first 10 slots): {prediction[0][:10]}")
prediction_actual = scaler.inverse_transform(prediction)
actual_target = scaler.inverse_transform(y_test[0:1])
print(f"\n   Predicted occupancy (rounded, first 10 slots): {np.round(prediction_actual[0][:10])}")
print(f"   Actual occupancy (first 10 slots): {actual_target[0][:10]}")
print("\n" + "=" * 60)
print("✅ LSTM Model Training Complete!")
print("=" * 60)
print(f"\n📁 Files created:")
print(f"   • {MODEL_PATH} - Trained LSTM model")
print(f"   • {SCALER_PATH} - Data scaler for preprocessing")
print(f"\n🚀 You can now use this model for real-time predictions!")
print(f"   Load with: model = keras.models.load_model('{MODEL_PATH}')")
