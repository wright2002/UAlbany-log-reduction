import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ----------------------- Config -----------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

TAB_EPOCHS = 10
HIST_EPOCHS = 10
TAB_BATCH = 256
HIST_BATCH = 64
TAB_LATENT = 4        # latent size for tabular MLP AE
HIST_LATENT = 2       # latent channels (after mean over time) for CNN AE
LR = 1e-3
HISTORY_STEPS = 12
HISTORY_CODES = [
    "ss", "h", "hh", "a", "aa", "d", "dd", "t", "tt", "c", "cc", "f", "ff",
    "r", "rr", "g", "caret", "w", "ww", "gg", "ii"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------- Load -------------------------
print("Loading data from Parquet...")
df = pd.concat([
    pd.read_parquet("training_data_1_1.parquet"),
    pd.read_parquet("training_data_3_1.parquet")
], ignore_index=True)
print("Data loaded.")

y = df["simple_label_encoded"].values.astype(np.int64)

# history columns (12 * 21 = 252)
hist_cols = [f"history_{i}_{code}" for i in range(1, HISTORY_STEPS + 1) for code in HISTORY_CODES]

# columns to drop from tabular view (labels + history one-hot)
drop_cols = [
    'simple_label_encoded',
    'label_c2c', 'label_filedownload', 'label_heartbeat', 'label_ddos',
    'label_okiru', 'label_torii', 'label_horizontal_scan', 'label_attack'
] + hist_cols

X_tab = df.drop(columns=drop_cols, errors="ignore")
X_hist = df[hist_cols].values.reshape(-1, HISTORY_STEPS, len(HISTORY_CODES)).astype(np.float32)

# scale tabular features
scaler = StandardScaler()
X_tab_scaled = scaler.fit_transform(X_tab).astype(np.float32)

# ------------------- Tabular MLP Autoencoder -------------------
class TabAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=TAB_LATENT):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        xhat = self.decoder(z)
        return xhat

tab_input_dim = X_tab_scaled.shape[1]
tab_model = TabAutoencoder(tab_input_dim, TAB_LATENT).to(device)
tab_criterion = nn.MSELoss()
tab_opt = torch.optim.Adam(tab_model.parameters(), lr=LR)

tab_ds = TensorDataset(torch.tensor(X_tab_scaled))
tab_loader = DataLoader(tab_ds, batch_size=TAB_BATCH, shuffle=True)

print("Training tabular MLP autoencoder...")
tab_model.train()
for epoch in range(TAB_EPOCHS):
    total = 0.0
    for (xb,) in tab_loader:
        xb = xb.to(device)
        tab_opt.zero_grad()
        recon = tab_model(xb)
        loss = tab_criterion(recon, xb)
        loss.backward()
        tab_opt.step()
        total += loss.item()
    print(f"[TAB] Epoch {epoch+1}/{TAB_EPOCHS}  MSE: {total/len(tab_loader):.6f}")
print("Tabular AE done.\n")

# ------------------- History 1D-CNN Autoencoder -------------------
class CNN1DAutoencoder(nn.Module):
    def __init__(self, latent_channels=HIST_LATENT):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(21, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(16, 8,  kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(8,  latent_channels, kernel_size=3, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_channels, 8,  kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(8, 16,  kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(16, 21, kernel_size=3, padding=1), nn.Sigmoid()
        )

    def forward(self, x):                 # x: [B, T=12, F=21]
        x = x.permute(0, 2, 1)            # -> [B, 21, 12]
        z = self.encoder(x)               # -> [B, C=latent, 12]
        xhat = self.decoder(z)            # -> [B, 21, 12]
        return xhat.permute(0, 2, 1)      # -> [B, 12, 21]

    def encode(self, x):                  # returns [B, latent]
        x = x.permute(0, 2, 1)            # [B, 21, 12]
        z = self.encoder(x)               # [B, C, 12]
        return torch.mean(z, dim=2)       # temporal mean -> [B, C]

hist_model = CNN1DAutoencoder(HIST_LATENT).to(device)
hist_criterion = nn.BCELoss()
hist_opt = torch.optim.Adam(hist_model.parameters(), lr=LR)

hist_ds = TensorDataset(torch.tensor(X_hist))
hist_loader = DataLoader(hist_ds, batch_size=HIST_BATCH, shuffle=True)

print("Training history CNN autoencoder...")
hist_model.train()
for epoch in range(HIST_EPOCHS):
    total = 0.0
    for (xb,) in hist_loader:
        xb = xb.to(device)
        hist_opt.zero_grad()
        preds = hist_model(xb)
        loss = hist_criterion(preds, xb)  # averaged BCE
        loss.backward()
        hist_opt.step()
        total += loss.item()
    print(f"[HIST] Epoch {epoch+1}/{HIST_EPOCHS}  BCE: {total/len(hist_loader):.6f}")
print("History AE done.\n")

# ------------------- Extract & Concatenate Latents -------------------
tab_model.eval()
hist_model.eval()
with torch.no_grad():
    Z_tab = tab_model.encoder(torch.tensor(X_tab_scaled, device=device)).cpu().numpy()          # [N, TAB_LATENT]
    Z_hist = hist_model.encode(torch.tensor(X_hist, device=device)).cpu().numpy()               # [N, HIST_LATENT]

Z = np.concatenate([Z_tab, Z_hist], axis=1)  # [N, TAB_LATENT + HIST_LATENT]

# ------------------- Train/Test Split & Classifier -------------------
X_train, X_test, y_train, y_test = train_test_split(
    Z, y, test_size=0.20, random_state=SEED, stratify=y
)

print("Training logistic regression on concatenated latents...")
t0 = time.time()
clf = LogisticRegression(max_iter=1000, solver='liblinear')
clf.fit(X_train, y_train)
print(f"Training completed in {time.time() - t0:.2f} seconds.\n")

y_pred = clf.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nF1 Score Report:")
print(classification_report(y_test, y_pred, digits=4))

# Optional: show absolute coefficients for quick interpretability
coef = pd.Series(clf.coef_[0], index=[*(f"tab_latent_{i}" for i in range(TAB_LATENT)),
                                      *(f"hist_latent_{i}" for i in range(HIST_LATENT))])
print("\nTop latent coefficients (abs):")
print(coef.abs().sort_values(ascending=False).head(15))
