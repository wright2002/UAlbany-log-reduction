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

EPOCHS = 10
LATENT_VECTOR_SIZE = 4

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

# === Define Autoencoder ===
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=LATENT_VECTOR_SIZE):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

# === Train Autoencoder ===
print("Training autoencoder...")
input_dim = X_scaled.shape[1]
autoencoder = Autoencoder(input_dim=input_dim, latent_dim=LATENT_VECTOR_SIZE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
dataset = TensorDataset(X_tensor)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

autoencoder.train()
for epoch in range(EPOCHS):
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

# === Get Latent Vectors ===
autoencoder.eval()
with torch.no_grad():
    X_latent = autoencoder.encoder(X_tensor).numpy()

# === Train/Test Split on Latent Vectors ===
X_train, X_test, y_train, y_test = train_test_split(
    X_latent, y, test_size=0.2, random_state=42, stratify=y
)

# === Logistic Regression on Latent Space ===
print("Training logistic regression on latent space...")
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

# === Feature Importances in Latent Space ===
coefficients = pd.Series(model.coef_[0], index=[f"latent_{i}" for i in range(LATENT_VECTOR_SIZE)])
top_features = coefficients.abs().sort_values(ascending=False).head(15)
print("Top 15 Latent Feature Importances:")
print(top_features)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_features.values, y=top_features.index)
plt.title("Top 15 Latent Feature Importances (LogReg)")
plt.xlabel("Absolute Coefficient Value")
plt.tight_layout()
plt.show()
