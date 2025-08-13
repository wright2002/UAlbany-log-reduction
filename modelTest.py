import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

print("[INFO] Loading data...")
df = pd.read_parquet("training_data_1_1.parquet")

# -----------------------------
# Set target label (already 0/1)
# -----------------------------
print("[DEBUG] Unique values in 'simple_label_encoded':", df['simple_label_encoded'].unique())
y = df['simple_label_encoded'].astype(int)

# -----------------------------
# Drop all label-related columns (prevent leakage)
# -----------------------------
label_related_cols = [
    'simple_label_encoded',  # true label
    'label_c2c', 'label_filedownload', 'label_heartbeat',
    'label_ddos', 'label_okiru', 'label_torii',
    'label_horizontal_scan', 'label_attack'
]
df.drop(columns=[col for col in label_related_cols if col in df.columns], inplace=True)

# -----------------------------
# Drop irrelevant/ID-based columns
# -----------------------------
drop_cols = ['uid', 'id.orig_h', 'id.resp_h', 'simple_label', 'label_source', 'id.orig_p', 'id.resp_p']
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

# -----------------------------
# Clean remaining data
# -----------------------------
df.drop(columns=df.columns[df.isna().mean() > 0.3], inplace=True)  # drop high-NaN cols
df.fillna(0, inplace=True)  # fill rest
df = df.select_dtypes(include=[np.number])  # keep only numeric

if df.empty:
    raise ValueError("[ERROR] DataFrame is still empty after cleanup")

X = df
print(f"[INFO] Final shape – X: {X.shape}, y: {y.shape}")

# -----------------------------
# Scale features
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# -----------------------------
# Build model
# -----------------------------
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# -----------------------------
# Train
# -----------------------------
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# -----------------------------
# Evaluate
# -----------------------------
y_pred = model.predict(X_test)
y_pred_labels = (y_pred > 0.5).astype(int)
acc = accuracy_score(y_test, y_pred_labels)
print(f"[INFO] Test Accuracy: {acc:.4f}")

# ✅ Save the trained model
model.save("final_model.h5")

# ✅ Save the fitted scaler
import joblib
joblib.dump(scaler, "final_scaler.pkl")

print("[INFO] Model and scaler saved.")
