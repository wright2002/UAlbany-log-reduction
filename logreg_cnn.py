import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ---------- Parameters ----------
BATCH_SIZE = 64
EPOCHS = 10
LATENT_DIM = 2

# === Load from Parquet ===
print("Loading data from Parquet...")
df = pd.concat([pd.read_parquet("training_data_1_1.parquet"), pd.read_parquet("training_data_3_1.parquet")])
print("Data loaded.")

# 21 one-hot codes from Zeek history
history_codes = [
    "ss", "h", "hh", "a", "aa", "d", "dd", "t", "tt", "c", "cc", "f", "ff",
    "r", "rr", "g", "caret", "w", "ww", "gg", "ii"
]

# Generate 252 column names
selected_cols = [f"history_{i}_{code}" for i in range(1, 13) for code in history_codes]

# Filter and reshape to [num_samples, 12, 21]
X = df[selected_cols].values.reshape(-1, 12, 21).astype(np.float32)

# Take the simple label field for testing
y = df["simple_label_encoded"].values.astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""
X_train = df1[selected_cols].values.reshape(-1, 12, 21).astype(np.float32)
X_test = df2[selected_cols].values.reshape(-1, 12, 21).astype(np.float32)
y_train = df1["simple_label_encoded"].values.astype(np.int64)
y_test = df2["simple_label_encoded"].values.astype(np.int64)
"""

train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(X_train)), batch_size=BATCH_SIZE, shuffle=True)

# ---------- Define Model ----------
class CNN1DAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(21, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, LATENT_DIM, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(LATENT_DIM, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 21, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)         # [B, 21, 12]
        latent = self.encoder(x)       # [B, LATENT_DIM, 12]
        recon = self.decoder(latent)   # [B, 21, 12]
        return recon.permute(0, 2, 1)  # [B, 12, 21]

    def encode(self, x):
        x = x.permute(0, 2, 1)         # [B, 21, 12]
        latent = self.encoder(x)       # [B, LATENT_DIM, 12]
        return torch.mean(latent, dim=2)  # Aggregate to [B, LATENT_DIM]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN1DAutoencoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

# ---------- Train Autoencoder with BCE (average loss) ----------
model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0
    batch_count = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)  # already averaged over batch
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        batch_count += 1
    avg_loss = epoch_loss / batch_count
    print(f"Epoch {epoch+1} Avg BCE Loss: {avg_loss:.6f}")

# ---------- Extract Latent Vectors ----------
model.eval()
with torch.no_grad():
    X_train_latent = model.encode(torch.tensor(X_train)).numpy()
    X_test_latent = model.encode(torch.tensor(X_test)).numpy()

# === Train Logistic Regression ===
print("Training logistic regression...")
start_time = time.time()
logreg = LogisticRegression(max_iter=1000, solver='liblinear')
logreg.fit(X_train_latent, y_train)
training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds.\n")

# === Evaluation ===
y_pred = logreg.predict(X_test_latent)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nF1 Score Report:")
print(classification_report(y_test, y_pred, digits=4))

# === Feature Importances ===
coefficients = pd.Series(logreg.coef_[0], index=[f"latent_{i}" for i in range(LATENT_DIM)])
top_features = coefficients.abs().sort_values(ascending=False).head(15)
print("Top 15 Feature Importances:")
print(top_features)
