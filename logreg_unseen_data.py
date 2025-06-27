import time
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# === Load Training and Test Data ===
print("Loading training data...")
train_df = pd.read_parquet("training_data_1.parquet")
print("Loading test data...")
test_df = pd.read_parquet("training_data_2.parquet")
print("Data loaded.\n")

# === Split features and labels ===
X_train = train_df.drop(columns=['simple_label_encoded'])
y_train = train_df['simple_label_encoded']

X_test = test_df.drop(columns=['simple_label_encoded'])
y_test = test_df['simple_label_encoded']

# === Scale based on training data ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Train Logistic Regression ===
print("Training logistic regression on full training set...")
start_time = time.time()
model = LogisticRegression(max_iter=1000, solver='liblinear')
model.fit(X_train_scaled, y_train)
training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds.\n")

# === Evaluate on test set ===
y_pred = model.predict(X_test_scaled)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nF1 Score Report:")
print(classification_report(y_test, y_pred, digits=4))

# === Feature Importances ===
coefficients = pd.Series(model.coef_[0], index=X_train.columns)
top_features = coefficients.abs().sort_values(ascending=False).head(15)
print("Top 15 Feature Importances:")
print(top_features)

# === Plot Feature Importances ===
plt.figure(figsize=(10, 6))
sns.barplot(x=top_features.values, y=top_features.index)
plt.title("Top 15 Feature Importances (Logistic Regression)")
plt.xlabel("Absolute Coefficient Value")
plt.tight_layout()
plt.show()
