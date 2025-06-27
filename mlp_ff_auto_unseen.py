import time
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# === Load Training and Test Data ===
print("Loading training data...")
train_df = pd.read_parquet("training_data_1.parquet")
print("Loading test data...")
test_df = pd.read_parquet("training_data_2.parquet")
print("Data loaded.\n")

X_train_raw = train_df.drop(columns=['simple_label_encoded'])
y_train = train_df['simple_label_encoded']

X_test_raw = test_df.drop(columns=['simple_label_encoded'])
y_test = test_df['simple_label_encoded']

# === Scale using training data only ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# === Define Autoencoder ===
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=20):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

# === Train Autoencoder ===
print("Training autoencoder...")
input_dim = X_train_scaled.shape[1]
autoencoder = Autoencoder(input_dim=input_dim, latent_dim=20)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
dataset = TensorDataset(X_train_tensor)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

autoencoder.train()
max_epochs = 20
for epoch in range(max_epochs):
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        recon = autoencoder(batch[0])
        loss = criterion(recon, batch[0])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
print("Autoencoder training complete.\n")

# === Encode Training and Test Data ===
autoencoder.eval()
with torch.no_grad():
    X_train_latent = autoencoder.encoder(X_train_tensor).numpy()
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    X_test_latent = autoencoder.encoder(X_test_tensor).numpy()

# === Train MLP Classifier on Latent Features ===
print("Training MLP classifier on latent features...")
start_time = time.time()
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32, 16),
    max_iter=300,
    activation='relu',
    solver='adam',
    random_state=42
)
mlp.fit(X_train_latent, y_train)
training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds.\n")

# === Evaluation on Unseen Latent Test Data ===
y_pred = mlp.predict(X_test_latent)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nF1 Score Report:")
print(classification_report(y_test, y_pred, digits=4))

# === Feature Importance via Input Layer Weights ===
print("Top 15 Latent Feature Weights (Input Layer):")
if hasattr(mlp, 'coefs_'):
    input_weights = np.abs(mlp.coefs_[0]).mean(axis=1)
    importance = pd.Series(input_weights, index=[f"latent_{i}" for i in range(X_train_latent.shape[1])])
    top_features = importance.sort_values(ascending=False).head(15)
    print(top_features)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_features.values, y=top_features.index)
    plt.title("Top 15 Latent Input Weights (MLP)")
    plt.xlabel("Mean Absolute Weight")
    plt.tight_layout()
    plt.show()
else:
    print("MLP does not expose input weights.")
