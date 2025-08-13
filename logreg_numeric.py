import time
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# === Load from Parquet ===
print("Loading data from Parquet...")
df = pd.concat([pd.read_parquet("training_data_1_1.parquet"), pd.read_parquet("training_data_3_1.parquet")])
print("Data loaded.")

X = df.drop(columns=[
    'simple_label_encoded',
    'label_c2c', 'label_filedownload', 'label_heartbeat', 'label_ddos',
    'label_okiru', 'label_torii', 'label_horizontal_scan', 'label_attack'
] + [f'history_{i}_{c.lower()}' for i in range(1, 13) for c in [
    'ss', 'h', 'hh', 'a', 'aa', 'd', 'dd', 't', 'tt', 'c', 'cc', 'f', 'ff',
    'r', 'rr', 'g', 'caret', 'w', 'ww', 'gg', 'ii'
]])
y = df['simple_label_encoded']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# === Train Logistic Regression ===
print("Training logistic regression...")
start_time = time.time()
model = LogisticRegression(max_iter=1000, solver='liblinear')
model.fit(X_train, y_train)
training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds.\n")

# === Evaluation ===
y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nF1 Score Report:")
print(classification_report(y_test, y_pred, digits=4))

# === Feature Importances ===
coefficients = pd.Series(model.coef_[0], index=X.columns)
top_features = coefficients.abs().sort_values(ascending=False).head(15)
print("Top 15 Feature Importances:")
print(top_features)

# Optional: Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=top_features.values, y=top_features.index)
plt.title("Top 15 Feature Importances (Logistic Regression)")
plt.xlabel("Absolute Coefficient Value")
plt.tight_layout()
plt.show()
