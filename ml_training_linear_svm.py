# ml_training_linear_svm.py
import argparse
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# ---- Default dataset path (change if needed) ----
DEFAULT_DATA_PATH = r"C:/Users/erikb/OneDrive/Desktop/BCYB 550/updates/training_data_1_1.parquet"

LEAKY_FEATURES = [
    "time_of_day_seconds", "duration",
    "orig_bytes", "resp_bytes", "missed_bytes",
    "orig_ip_bytes", "resp_ip_bytes",
    "history_1_ss", "history_1_a", "history_1_d",
    "simple_label_encoded",
    "binary_label",
]

def build_labels_and_features(df: pd.DataFrame):
    # Build binary_label from label_* columns
    label_cols = [c for c in df.columns if c.startswith("label_")]
    attack_cols = [c for c in label_cols if c != "label_attack"]
    if not label_cols:
        raise ValueError("No columns starting with 'label_' were found in the dataset.")
    df["binary_label"] = df[attack_cols].any(axis=1).astype(int)
    y = df["binary_label"].astype(int)

    # Drop label_* + leaky fields from X
    X = df.drop(columns=label_cols + LEAKY_FEATURES, errors="ignore")

    print(f"Found {len(label_cols)} label_* columns")
    print("Class counts (binary_label):")
    print(y.value_counts())
    print(f"Feature count after drops: {X.shape[1]}")
    return X, y

def print_metrics(name, y_true, y_pred):
    print(f"\n=== {name} ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred, digits=4))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA_PATH,
                    help="Path to parquet with label_* columns (defaults to your usual path).")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    print(f"Loading dataset from: {args.data}")
    df = pd.read_parquet(args.data)
    print("Dataset shape:", df.shape)

    X, y = build_labels_and_features(df)

    # Stratified split (keep natural imbalance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )

    # ===== Shared scaler for LR / SVM =====
    shared_scaler = StandardScaler()
    X_train_shared = shared_scaler.fit_transform(X_train)
    X_test_shared = shared_scaler.transform(X_test)

    # --- Logistic Regression ---
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_train_shared, y_train)
    y_pred_lr = lr.predict(X_test_shared)
    print_metrics("Logistic Regression", y_test, y_pred_lr)
    joblib.dump(lr, "logistic_regression.pkl")

    # --- Random Forest (unscaled features) ---
    rf = RandomForestClassifier(
        class_weight="balanced", n_estimators=300, random_state=args.random_state, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    print_metrics("Random Forest", y_test, y_pred_rf)
    joblib.dump(rf, "random_forest.pkl")

    # --- FAST SVM: Calibrated LinearSVC (probabilities) ---
    base = LinearSVC(C=1.0, class_weight="balanced", random_state=args.random_state)
    svm = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    svm.fit(X_train_shared, y_train)
    y_pred_svm = svm.predict(X_test_shared)
    print_metrics("Support Vector Machine (Calibrated LinearSVC)", y_test, y_pred_svm)
    joblib.dump(svm, "support_vector_machine.pkl")

    # Save shared scaler & features for your evaluator
    joblib.dump(shared_scaler, "scaler.pkl")
    with open("features.txt", "w") as f:
        for col in X.columns:
            f.write(f"{col}\n")

    # ===== MLP path: RobustScaler + manual class balancing (NO SMOTE) =====
    print("\n=== MLP Classifier (RobustScaler + manual balancing, no SMOTE) ===")
    mlp_scaler = RobustScaler()
    X_train_mlp = mlp_scaler.fit_transform(X_train)
    X_test_mlp = mlp_scaler.transform(X_test)

    # Upsample minority class in TRAIN ONLY (balance without synthetic interpolation)
    train_df = pd.DataFrame(X_train_mlp)
    train_df["y"] = y_train.values
    n_pos = (train_df["y"] == 1).sum()
    n_neg = (train_df["y"] == 0).sum()

    if n_pos == 0 or n_neg == 0:
        # Edge case guard
        balanced_df = train_df.copy()
    elif n_pos < n_neg:
        pos = train_df[train_df["y"] == 1]
        neg = train_df[train_df["y"] == 0]
        pos_up = pos.sample(len(neg), replace=True, random_state=args.random_state)
        balanced_df = pd.concat([neg, pos_up], ignore_index=True)
    else:
        pos = train_df[train_df["y"] == 1]
        neg = train_df[train_df["y"] == 0]
        neg_up = neg.sample(len(pos), replace=True, random_state=args.random_state)
        balanced_df = pd.concat([pos, neg_up], ignore_index=True)

    X_train_bal = balanced_df.drop(columns="y").values
    y_train_bal = balanced_df["y"].values

    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        max_iter=1000,
        early_stopping=True,
        learning_rate_init=1e-4,
        alpha=1e-3,
        random_state=args.random_state,
        verbose=False,
    )
    mlp.fit(X_train_bal, y_train_bal)
    y_pred_mlp = mlp.predict(X_test_mlp)
    print_metrics("MLP Classifier (balanced)", y_test, y_pred_mlp)

    try:
        y_prob_mlp = mlp.predict_proba(X_test_mlp)[:, 1]
        auc = roc_auc_score(y_test, y_prob_mlp)
        print(f"MLP ROC-AUC: {auc:.4f}")
    except Exception as e:
        print(f"MLP ROC-AUC unavailable: {e}")

    joblib.dump(mlp, "mlp_classifier.pkl")
    joblib.dump(mlp_scaler, "mlp_scaler.pkl")  # optional if you ever eval MLP separately

    print("\n✅ Saved models: logistic_regression.pkl, random_forest.pkl, support_vector_machine.pkl, mlp_classifier.pkl")
    print("✅ Saved scalers: scaler.pkl (LR/RF/SVM), mlp_scaler.pkl (MLP)")
    print("✅ Saved features.txt")

if __name__ == "__main__":
    main()
