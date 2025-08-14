
import argparse
import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve,
    f1_score, precision_score, recall_score, roc_auc_score
)

def load_models():
    candidates = {
        "Logistic Regression": "logistic_regression.pkl",
        "MLP Classifier": "mlp_classifier.pkl",
        "Support Vector Machine": "support_vector_machine.pkl",
        "Random Forest": "random_forest.pkl",
    }
    models = {}
    for name, fname in candidates.items():
        if os.path.exists(fname):
            models[name] = joblib.load(fname)
    if not models:
        raise FileNotFoundError("No model .pkl files found in the working directory.")
    return models

def get_expected_features():
    if not os.path.exists("features.txt"):
        raise FileNotFoundError("features.txt not found. Make sure you ran the training script and saved feature names.")
    with open("features.txt", "r") as f:
        expected = [line.strip() for line in f.readlines() if line.strip()]
    return expected

def build_binary_label(df):
    label_cols = [c for c in df.columns if c.startswith("label_")]
    attack_cols = [c for c in label_cols if c != "label_attack"]
    df["binary_label"] = df[attack_cols].any(axis=1).astype(int)
    return df, label_cols

def drop_leaky_and_labels(df, label_cols):
    leaky_features = [
        "time_of_day_seconds", "duration",
        "orig_bytes", "resp_bytes", "missed_bytes",
        "orig_ip_bytes", "resp_ip_bytes",
        "history_1_ss", "history_1_a", "history_1_d"
    ]
    drop_cols = list(set(label_cols + leaky_features + ["binary_label", "simple_label_encoded"]))
    return df.drop(columns=drop_cols, errors="ignore")

def align_and_scale(df, expected_features, scaler):
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0
    X = df[expected_features]
    Xs = scaler.transform(X)
    return Xs

def proba_or_score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        s_min, s_max = scores.min(), scores.max()
        if s_max == s_min:
            return np.full_like(scores, 0.5, dtype=float)
        return (scores - s_min) / (s_max - s_min)
    else:
        return model.predict(X).astype(float)

def evaluate_at_threshold(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    bal_acc = (recall + specificity) / 2.0
    return {
        "threshold": thr,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "precision": precision, "recall": recall, "specificity": specificity,
        "f1": f1, "balanced_accuracy": bal_acc
    }

def choose_threshold(y_true, y_prob, strategy="f1",
                     min_specificity=None, min_precision=None, min_recall=None):
    thresholds = np.linspace(0.0, 1.0, 101)
    metrics = [evaluate_at_threshold(y_true, y_prob, t) for t in thresholds]

    # Apply floors (filters)
    filtered = []
    for m in metrics:
        if (min_specificity is not None and m["specificity"] < min_specificity):
            continue
        if (min_precision is not None and m["precision"] < min_precision):
            continue
        if (min_recall is not None and m["recall"] < min_recall):
            continue
        filtered.append(m)

    if filtered:
        pool = filtered
        used_filter = True
    else:
        pool = metrics
        used_filter = False

    if strategy == "f1":
        best = max(pool, key=lambda m: m["f1"])
    elif strategy == "f2":
        def f2(m):
            p, r = m["precision"], m["recall"]
            return (5 * p * r) / (4 * p + r) if (4 * p + r) else 0.0
        best = max(pool, key=f2)
    elif strategy == "youden":
        best = max(pool, key=lambda m: m["recall"] + m["specificity"] - 1)
    else:
        raise ValueError("Unknown strategy. Use one of: f1, f2, youden")
    return best, metrics, used_filter

def main():
    parser = argparse.ArgumentParser(description="Predict with adjustable thresholds and evaluate.")
    parser.add_argument("--data", default="training_data_34_1.parquet",
                        help="Parquet file with labeled samples for evaluation.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Fixed decision threshold (0..1). If omitted, optimize per --optimize.")
    parser.add_argument("--optimize", choices=["f1", "f2", "youden"], default="f1",
                        help="Strategy to pick threshold when --threshold is not provided.")
    parser.add_argument("--min_specificity", type=float, default=None,
                        help="Minimum required specificity (TNR) when searching thresholds.")
    parser.add_argument("--min_precision", type=float, default=None,
                        help="Minimum required precision when searching thresholds.")
    parser.add_argument("--min_recall", type=float, default=None,
                        help="Minimum required recall (TPR) when searching thresholds.")
    parser.add_argument("--save_csv", default="threshold_scan_metrics.csv",
                        help="Where to save per-threshold metrics (appends per model).")
    parser.add_argument("--save_reports_dir", default="model_reports",
                        help="Directory to save per-model classification reports and confusion matrices.")
    parser.add_argument("--export_predictions_dir", default=None,
                        help="If set, export per-model predictions CSV with y_true, y_prob, y_pred.")
    args = parser.parse_args()

    scaler = joblib.load("scaler.pkl")
    expected_features = get_expected_features()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data file not found: {args.data}")
    df = pd.read_parquet(args.data)
    df, label_cols = build_binary_label(df)
    y_true = df["binary_label"].astype(int).values

    X_raw = drop_leaky_and_labels(df.copy(), label_cols)
    X = align_and_scale(X_raw, expected_features, scaler)

    models = load_models()
    os.makedirs(args.save_reports_dir, exist_ok=True)
    if args.export_predictions_dir:
        os.makedirs(args.export_predictions_dir, exist_ok=True)

    all_rows = []
    for name, model in models.items():
        print(f"\n=== {name} ===")
        y_prob = proba_or_score(model, X)

        if args.threshold is not None:
            chosen = evaluate_at_threshold(y_true, y_prob, args.threshold)
            chosen_strat = f"fixed@{args.threshold:.2f}"
            used_filter = False
            metrics = None
        else:
            chosen, metrics, used_filter = choose_threshold(
                y_true, y_prob, strategy=args.optimize,
                min_specificity=args.min_specificity,
                min_precision=args.min_precision,
                min_recall=args.min_recall
            )
            chosen_strat = args.optimize
            if used_filter:
                chosen_strat += " +floors"

            if metrics:
                for m in metrics:
                    row = {"model": name, "strategy": args.optimize,
                           "min_specificity": args.min_specificity,
                           "min_precision": args.min_precision,
                           "min_recall": args.min_recall,
                           **m}
                    all_rows.append(row)

        thr = chosen["threshold"]
        summary = {k: v for k, v in chosen.items() if k not in ('tp','fp','tn','fn')}
        print(f"Chosen threshold ({chosen_strat}): {thr:.3f}")
        print(json.dumps(summary, indent=2))

        y_pred = (y_prob >= thr).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        print("\nConfusion matrix [tn, fp; fn, tp]:")
        print(cm)

        print("\nClassification report:")
        report = classification_report(y_true, y_pred, digits=4)
        print(report)

        base = name.lower().replace(" ", "_")
        with open(os.path.join(args.save_reports_dir, f"{base}_report.txt"), "w") as f:
            f.write(f"{name}\n")
            f.write(f"Threshold strategy: {chosen_strat}\n")
            f.write(f"Chosen threshold: {thr:.4f}\n\n")
            f.write("Confusion matrix [tn, fp; fn, tp]:\n")
            f.write(np.array2string(cm))
            f.write("\n\nClassification report:\n")
            f.write(report)

        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        pr_df = pd.DataFrame({"precision": precision, "recall": recall})
        pr_df.to_csv(os.path.join(args.save_reports_dir, f"{base}_pr_points.csv"), index=False)

        if args.export_predictions_dir:
            out_df = pd.DataFrame({
                "y_true": y_true,
                "y_prob": y_prob,
                "y_pred": y_pred
            })
            out_df.to_csv(os.path.join(args.export_predictions_dir, f"{base}_predictions.csv"), index=False)

        summary_row = {
            "model": name,
            "strategy": chosen_strat,
            "min_specificity": args.min_specificity,
            "min_precision": args.min_precision,
            "min_recall": args.min_recall,
            **chosen
        }
        all_rows.append(summary_row)

        try:
            auc = roc_auc_score(y_true, y_prob)
            print(f"ROC-AUC: {auc:.4f}")
        except Exception as e:
            print(f"ROC-AUC unavailable: {e}")

    if all_rows:
        df_csv = pd.DataFrame(all_rows)
        if os.path.exists(args.save_csv):
            prev = pd.read_csv(args.save_csv)
            df_csv = pd.concat([prev, df_csv], ignore_index=True)
        df_csv.to_csv(args.save_csv, index=False)
        print(f"\nSaved per-threshold metrics to: {args.save_csv}")
        print("You can filter by 'model' and 'strategy'.")

    print("\n✅ Done.")

if __name__ == "__main__":
    main()
