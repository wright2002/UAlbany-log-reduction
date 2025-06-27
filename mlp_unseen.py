import time
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
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

X_train = train_df.drop(columns=['simple_label_encoded'])
y_train = train_df['simple_label_encoded']

X_test = test_df.drop(columns=['simple_label_encoded'])
y_test = test_df['simple_label_encoded']

# === Scale Using Training Data ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

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
mlp.fit(X_train_scaled, y_train)
training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds.\n")

# === Evaluation on Test Data ===
y_pred = mlp.predict(X_test_scaled)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nF1 Score Report:")
print(classification_report(y_test, y_pred, digits=4))

# === Feature Importance via Input Layer Weights ===
print("Top 15 Feature Weights (Input Layer):")
if hasattr(mlp, 'coefs_'):
    input_weights = np.abs(mlp.coefs_[0]).mean(axis=1)
    importance = pd.Series(input_weights, index=X_train.columns)
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
