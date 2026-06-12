"""
Produces all required metrics: precision/recall/F1, OOS rejection rate,
false rejection rate, confusion matrix (for evaluvating classification performance of the model)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from inference import CommandClassifier
import json

def main():
    # Load test data
    df = pd.read_csv("data/dataset.csv")
    df_test = df[df["split"] == "test"]

    # Initialise classifier
    classifier = CommandClassifier(
        onnx_model_path="models/model_quantized.onnx",
        prototypes_path="models/prototypes.npy",
        labels_path="models/class_labels.json",
        threshold_path="models/threshold.json"
    )

    # Run predictions
    y_true = []
    y_pred = []
    for _, row in df_test.iterrows():
        text = row["text"]
        true_label = row["label"]
        pred_label, _ = classifier.predict(text)
        y_true.append(true_label)
        y_pred.append(pred_label)

    # In‑scope classification report
    # Filter out OOS from both true and pred (for report)
    inscope_true = [t for t in y_true if t != "oos"]
    inscope_pred = [p for p, t in zip(y_pred, y_true) if t != "oos"]

    
    print("CLASSIFICATION REPORT (In-scope commands only)")
    print('---------------------------------------------------')
    print(classification_report(inscope_true, inscope_pred, zero_division=0))

    # OOS rejection rate
    oos_indices = [i for i, lbl in enumerate(y_true) if lbl == "oos"]
    oos_pred_labels = [y_pred[i] for i in oos_indices]
    oos_rejected = sum(1 for p in oos_pred_labels if p == "reject")
    oos_rejection_rate = oos_rejected / len(oos_indices) if oos_indices else 0.0
    print(f"\nOOS rejection rate: {oos_rejection_rate:.3f} ({oos_rejected}/{len(oos_indices)})")

    # False rejection rate (valid commands rejected)
    inscope_indices = [i for i, lbl in enumerate(y_true) if lbl != "oos"]
    inscope_pred_labels = [y_pred[i] for i in inscope_indices]
    false_rejects = sum(1 for p in inscope_pred_labels if p == "reject")
    false_rejection_rate = false_rejects / len(inscope_indices) if inscope_indices else 0.0
    print(f"False rejection rate: {false_rejection_rate:.3f} ({false_rejects}/{len(inscope_indices)})")

    # Confusion matrix (only in-scope classes) - to evaluate classification performance of the model
    unique_labels = sorted(set(inscope_true))
    cm = confusion_matrix(inscope_true, inscope_pred, labels=unique_labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=unique_labels, yticklabels=unique_labels, cmap="Blues")
    plt.title("Confusion Matrix - Command Classes")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("\nConfusion matrix saved as confusion_matrix.png")

if __name__ == "__main__":
    main()