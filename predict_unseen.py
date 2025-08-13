import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from sklearn.metrics import accuracy_score

print("[INFO] Loading saved model and scaler...")
model = load_model("final_model.h5")
scaler = joblib.load("final_scaler.pkl")

print("[INFO] Loading unseen data...")
df = pd.read_parquet("benign_data_7_1.parquet")  # <-- Change filename if needed

# Optional: capture labels if they exist
true_labels = None
if 'simple_label_encoded' in df.columns:
    true_labels = df['simple_label_encoded'].astype(int).values

# Drop label and ID columns (same as in training)
label_cols = [
    'simple_label_encoded',
    'label_c2c', 'label_filedownload', 'label_heartbeat',
    'label_ddos', 'label_okiru', 'label_torii',
    'label_horizontal_scan', 'label_attack'
]
id_cols = ['uid', 'id.orig_h', 'id.resp_h', 'simple_label', 'label_source', 'id.orig_p', 'id.resp_p']
df.drop(columns=[col for col in label_cols + id_cols if col in df.columns], inplace=True, errors='ignore')

# Clean data
df = df.select_dtypes(include=[np.number])
df.fillna(0, inplace=True)

# Scale it
print("[INFO] Scaling unseen data...")
X_unseen = scaler.transform(df)

# Predict
print("[INFO] Predicting...")
predictions = model.predict(X_unseen)
pred_labels = (predictions > 0.5).astype(int).flatten()

# Save to CSV
pd.DataFrame({'prediction': pred_labels}).to_csv("benign7.csv", index=False)
print("[INFO] Predictions saved to unseen_predictions.csv")

# Accuracy if labels are present
if true_labels is not None:
    acc = accuracy_score(true_labels, pred_labels)
    print(f"[INFO] Accuracy on unseen data: {acc:.4f}")
else:
    print("[INFO] No true labels found — skipping accuracy check.")
