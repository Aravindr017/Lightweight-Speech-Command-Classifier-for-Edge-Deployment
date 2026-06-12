"""
Loads the dataset, uses a sentence-transformer model to compute class prototypes,
and selects an optimal OOS rejection threshold based on validation data. (using cosine similarity + class prototypes)
"""

import pandas as pd
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple

def compute_prototypes(
    model: SentenceTransformer,
    df_train: pd.DataFrame
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    For each command class, average the embeddings of its training sentences.
    Return:
        prototypes: dict {label: embedding_vector} ( as dict)
        class_labels: sorted list of all labels (excluding 'oos')
    """
    # Filter only training samples that are not out‑of‑scope
    train_mask = (df_train["split"] == "train") & (df_train["label"] != "oos")
    train_df = df_train[train_mask]

    sentences = train_df["text"].tolist()
    labels = train_df["label"].tolist()

    # Compute all embeddings at once 
    embeddings = model.encode(sentences, show_progress_bar=True)

    unique_labels = sorted(set(labels))
    prototypes = {}
    for lbl in unique_labels:
        # Find indices where label equals lbl
        indices = [i for i, l in enumerate(labels) if l == lbl]
        class_embeds = embeddings[indices]          # shape (n_samples, embed_dim)
        prototypes[lbl] = np.mean(class_embeds, axis=0)   # centroid
    return prototypes, unique_labels

def find_optimal_threshold(
    model: SentenceTransformer,
    prototypes: Dict[str, np.ndarray],
    df_val: pd.DataFrame
) -> float:
    """
    Evaluate candidate thresholds on validation set (contains in-scope and OOS).
    Choose threshold that maximises F1 score for in-scope classification.
    """
    # Split validation into in‑scope and OOS
    val_inscope = df_val[(df_val["split"] == "val") & (df_val["label"] != "oos")]
    val_ood = df_val[(df_val["split"] == "val") & (df_val["label"] == "oos")]

    # For each in‑scope sample, compute similarity to its correct prototype
    correct_similarities = []
    for _, row in val_inscope.iterrows():
        text = row["text"]
        true_label = row["label"]
        emb = model.encode([text])[0]
        correct_proto = prototypes[true_label]
        sim = cosine_similarity([emb], [correct_proto])[0][0]   # cosine similarity
        correct_similarities.append(sim)

    # For each OOS sample, compute maximum similarity to ANY prototype
    max_ood_similarities = []
    for _, row in val_ood.iterrows():
        text = row["text"]
        emb = model.encode([text])[0]
        sims = [cosine_similarity([emb], [prototypes[lbl]])[0][0] for lbl in prototypes]
        max_ood_similarities.append(max(sims))

        # finding best threshold value (for cosine similarity checking)
    # Try thresholds from 0.4 to 0.9 as setting step as 0.02
    best_f1 = 0.0
    best_threshold = 0.6
    for th in np.arange(0.4, 0.91, 0.02):
        # True positive: in‑scope (similarity >= th)
        tp = sum(1 for s in correct_similarities if s >= th)
        # False negative: in‑scope (similarity < th)
        fn = len(correct_similarities) - tp
        # True negative: OOS  (max_sim < th)
        tn = sum(1 for s in max_ood_similarities if s < th)
        # False positive: OOS (max_sim >= th)
        fp = len(max_ood_similarities) - tn

        if tp + fp == 0 or tp + fn == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = th

    return best_threshold

def main():
    df = pd.read_csv("data/dataset.csv")
    print(f"Loaded dataset with {len(df)} rows")

    # Load pre‑trained sentence transformer (this model is small: ~80MB in FP32, but we will quantise to 8-bit integer (INT8) to reduce size and speed up inference(prediction))
    model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
    print("Model loaded.")

    # Compute prototypes using training data    (classify the commands based on similarity to class prototypes)
    prototypes, class_labels = compute_prototypes(model, df)
    print(f"Computed prototypes for {len(class_labels)} classes: {class_labels}")

    # Find optimal threshold using validation set
    threshold = find_optimal_threshold(model, prototypes, df)
    print(f"Optimal threshold = {threshold:.3f}")

    # Save (numpy, json)
    # Convert prototypes dict to a list in the same order as class_labels
    prototype_list = [prototypes[lbl] for lbl in class_labels]
    np.save("models/prototypes.npy", np.array(prototype_list))
    with open("models/class_labels.json", "w") as f:
        json.dump(class_labels, f)
    with open("models/threshold.json", "w") as f:
        json.dump({"threshold": float(threshold)}, f)

    print("Training completed, prototypes and threshold saved to models/ directory.")

if __name__ == "__main__":
    main()