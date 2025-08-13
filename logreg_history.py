import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# === Load from Parquet ===
print("Loading data from Parquet...")
df = pd.read_parquet("training_data_1.parquet")
print("Data loaded.")

# 21 one-hot codes from Zeek history
history_codes = [
    "ss", "h", "hh", "a", "aa", "d", "dd", "t", "tt", "c", "cc", "f", "ff",
    "r", "rr", "g", "caret", "w", "ww", "gg", "ii"
]

# Generate 252 column names
selected_cols = [f"history_{i}_{code}" for i in range(1, 13) for code in history_codes]

# Filter and reshape to [num_samples, 12, 21]
X = df[selected_cols]

# Take the simple label field for testing
y = df["simple_label_encoded"].values.astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train Logistic Regression ===
print("Training logistic regression...")
start_time = time.time()
logreg = LogisticRegression(max_iter=1000, solver='liblinear')
logreg.fit(X_train, y_train)
training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds.\n")

# === Evaluation ===
y_pred = logreg.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nF1 Score Report:")
print(classification_report(y_test, y_pred, digits=4))

# === Feature Importances ===
coefficients = pd.Series(logreg.coef_[0], index=X.columns)
top_features = coefficients.abs().sort_values(ascending=False).head(15)
print("Top 15 Feature Importances:")
print(top_features)
