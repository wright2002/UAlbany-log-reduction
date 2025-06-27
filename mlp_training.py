import time
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# === Load from Parquet ===
print("Loading data from Parquet...")
df = pd.read_parquet("training_data_1.parquet")
print("Data loaded.")

# === Split features and target ===
X = df.drop(columns=['simple_label_encoded'])
y = df['simple_label_encoded']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# === Train MLP Classifier ===
print("Training MLP classifier...")
start_time = time.time()
mlp = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128, 64),
    max_iter=300,
    activation='relu',
    solver='adam',
    random_state=42
)
mlp.fit(X_train, y_train)
training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds.\n")

# === Evaluation ===
y_pred = mlp.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nF1 Score Report:")
print(classification_report(y_test, y_pred, digits=4))

# === Optional: Feature importance via absolute weights of input layer ===
print("Top 15 Feature Weights (Input Layer):")
if hasattr(mlp, 'coefs_'):
    input_weights = np.abs(mlp.coefs_[0]).mean(axis=1)
    importance = pd.Series(input_weights, index=X.columns)
    top_features = importance.sort_values(ascending=False).head(15)
    print(top_features)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_features.values, y=top_features.index)
    plt.title("Top 15 Input Feature Weights (MLP)")
    plt.xlabel("Mean Absolute Weight")
    plt.tight_layout()
    plt.show()
else:
    print("MLP does not expose input weights.")
