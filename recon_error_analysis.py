import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import csv

DATAFILE = "training_data_1_1"
EPOCHS = 10
LATENT_VECTOR_SIZE = 8

# === Load and Scale Data ===
print("Loading data from Parquet...")
df = pd.read_parquet(DATAFILE + ".parquet")
print("Data loaded.")

X = df.drop(columns=[
    'simple_label_encoded',
    'label_c2c', 'label_filedownload', 'label_heartbeat', 'label_ddos',
    'label_okiru', 'label_torii', 'label_horizontal_scan', 'label_attack'
] + [f'history_{i}_{c.lower()}' for i in range(1, 13) for c in [
    'ss', 'h', 'hh', 'a', 'aa', 'd', 'dd', 't', 'tt', 'c', 'cc', 'f', 'ff',
    'r', 'rr', 'g', 'caret', 'w', 'ww', 'gg', 'ii'
]])


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Define Autoencoder ===
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=LATENT_VECTOR_SIZE):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

# === Train Autoencoder ===
print("Training autoencoder...")
input_dim = X_scaled.shape[1]
autoencoder = Autoencoder(input_dim=input_dim, latent_dim=20)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
dataset = TensorDataset(X_tensor)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# Prepare CSV output
columns = X.columns.tolist()
error_log = []

# Updated training loop
print("Training autoencoder with per-column reconstruction error tracking...")
for epoch in range(EPOCHS):
    epoch_errors = np.zeros(X_scaled.shape[1])
    total_samples = 0
    total_loss = 0

    for batch in dataloader:
        optimizer.zero_grad()
        input_batch = batch[0]
        recon_batch = autoencoder(input_batch)
        loss = criterion(recon_batch, input_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Per-feature squared error
        errors = (recon_batch.detach().numpy() - input_batch.numpy()) ** 2
        epoch_errors += errors.sum(axis=0)
        total_samples += input_batch.size(0)

    avg_loss = total_loss / len(dataloader)
    mean_errors = epoch_errors / total_samples
    error_log.append(mean_errors.tolist())
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# Write to CSV
with open(DATAFILE + "recon_err.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch"] + columns)
    for i, row in enumerate(error_log):
        writer.writerow([i + 1] + row)

print("Reconstruction error log saved.")
