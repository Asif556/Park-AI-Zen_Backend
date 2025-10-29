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
print("=" * 70)
print("🚗 LSTM Initial Model Training with 1000 Synthetic Samples")
print("=" * 70)
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
print(f"✅ Fetched {len(real_sessions)} real sessions from database")
print(f"\n[2/8] Generating {SYNTHETIC_SAMPLES} realistic synthetic samples...")
synthetic_sessions = []
base_time = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
base_time = base_time - timedelta(days=30)
patterns = {
    'morning_rush': {
        'hours': list(range(7, 11)),
        'slots': list(range(1, 21)),
        'duration_range': (60, 240),
        'occupancy_range': (12, 20),
        'probability': 0.25
    },
    'midday': {
        'hours': list(range(11, 15)),
        'slots': list(range(1, 31)),
        'duration_range': (30, 120),
        'occupancy_range': (6, 12),
        'probability': 0.20
    },
    'afternoon': {
        'hours': list(range(14, 18)),
        'slots': list(range(1, 26)),
        'duration_range': (45, 180),
        'occupancy_range': (8, 15),
        'probability': 0.20
    },
    'evening_rush': {
        'hours': list(range(17, 21)),
        'slots': list(range(1, 21)),
        'duration_range': (60, 300),
        'occupancy_range': (10, 18),
        'probability': 0.25
    },
    'night': {
        'hours': list(range(0, 7)) + list(range(20, 24)),
        'slots': list(range(1, 16)),
        'duration_range': (120, 480),
        'occupancy_range': (2, 8),
        'probability': 0.10
    }
}
samples_per_pattern = {
    pattern: int(SYNTHETIC_SAMPLES * config['probability'])
    for pattern, config in patterns.items()
}
for pattern_name, num_samples in samples_per_pattern.items():
    pattern = patterns[pattern_name]
    for i in range(num_samples):
        day_offset = random.randint(0, 29)
        hour = random.choice(pattern['hours'])
        minute = random.choice([0, 15, 30, 45])
        entry_time = base_time + timedelta(days=day_offset, hours=hour, minutes=minute)
        is_weekend = entry_time.weekday() >= 5
        if is_weekend:
            occupancy = random.randint(
                max(1, pattern['occupancy_range'][0] - 3),
                pattern['occupancy_range'][1] - 2
            )
        else:
            occupancy = random.randint(*pattern['occupancy_range'])
        duration_minutes = random.randint(*pattern['duration_range'])
        exit_time = entry_time + timedelta(minutes=duration_minutes)
        slot_id = random.choice(pattern['slots'])
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
print(f"   Pattern distribution:")
for pattern_name, count in samples_per_pattern.items():
    print(f"   • {pattern_name}: {count} samples")
print(f"\n[3/8] Combining real and synthetic data...")
all_sessions = real_sessions + synthetic_sessions
print(f"✅ Total training samples: {len(all_sessions)}")
print(f"   • Real: {len(real_sessions)}")
print(f"   • Synthetic: {len(synthetic_sessions)}")
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
    print("❌ No sequences created! Need more data.")
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
    test_size=0.15,
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
    metrics=['accuracy', 'binary_accuracy', 'mse']
)
print(model.summary())
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)
model_checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)
print("\n🏋️ Training model... (this may take a few minutes)")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, model_checkpoint, reduce_lr],
    verbose=1
)
test_loss, test_accuracy, test_binary_acc, test_mse = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Loss: {test_loss:.4f}")
print(f"✅ Test Accuracy (exact match): {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"✅ Test Binary Accuracy (per slot): {test_binary_acc:.4f} ({test_binary_acc*100:.2f}%)")
print(f"✅ Test MSE: {test_mse:.4f}")
model.save(MODEL_PATH)
print(f"✅ Model saved to '{MODEL_PATH}'")
sample_input = X_test[0:5]
predictions = model.predict(sample_input, verbose=0)
predictions_actual = scaler.inverse_transform(predictions)
actual_targets = scaler.inverse_transform(y_test[0:5])
print(f"\n🧪 Sample predictions (first 5 samples, first 10 slots):")
for i in range(5):
    pred_rounded = np.round(predictions_actual[i][:10])
    actual = actual_targets[i][:10]
    matches = np.sum(pred_rounded == actual)
    print(f"   Sample {i+1}: Predicted={pred_rounded} | Actual={actual} | Matches={matches}/10")
avg_occupancy = np.mean(predictions_actual) * 100
print(f"\n📊 Average predicted occupancy: {avg_occupancy:.1f}%")
print("\n" + "=" * 70)
print("✅ Initial LSTM Model Training Complete!")
print("=" * 70)
print(f"\n📊 Training Summary:")
print(f"   • Initial synthetic samples: {len(synthetic_sessions)}")
print(f"   • Real sessions included: {len(real_sessions)}")
print(f"   • Total training sequences: {len(X)}")
print(f"   • Binary accuracy (per slot): {test_binary_acc*100:.2f}%")
print(f"   • Model file: {MODEL_PATH}")
print(f"   • Scaler file: {SCALER_PATH}")
print(f"\n💡 Next Steps:")
print(f"   1. Restart Flask: python app.py")
print(f"   2. Real data will automatically be added to training on every 5th exit")
print(f"   3. Model will continuously improve with real usage patterns")
print(f"\n🚀 Base model is ready! Real data will enhance it over time.")
