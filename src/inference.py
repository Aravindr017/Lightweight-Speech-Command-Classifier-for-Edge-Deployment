"""
Provides a CommandClassifier class that loads the ONNX model (INT8 quantized),
prototypes, labels, and threshold. Uses ONNX Runtime for fast, offline inference.
"""

import numpy as np
import json
import onnxruntime as ort
from transformers import AutoTokenizer
from typing import Tuple, List
import os

class CommandClassifier:
    
    """
    Lightweight semantic command classifier for edge deployment.
    Uses ONNX Runtime with INT8 quantized model.
    """
    
    def __init__(
        self,
        onnx_model_path: str,
        prototypes_path: str,
        labels_path: str,
        threshold_path: str,
        tokenizer_name: str = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    ):
        # Check if model file exists
        if not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")
        
        # Load ONNX session (CPU execution) - correct way: first argument is the path
        self.session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
        
        # Load prototypes (list of embedding vectors)
        self.prototypes = np.load(prototypes_path)   # shape (num_classes, embedding_dim)
        
        # Load class labels (list of strings)
        with open(labels_path, "r") as f:
            self.labels: List[str] = json.load(f)
        
        # Load threshold (float)
        with open(threshold_path, "r") as f:
            self.threshold: float = json.load(f)["threshold"]

        # Load tokenizer (the same one used during export)
        # This will be cached locally after first download – offline afterwards.
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Convert input text to a sentence embedding vector using the ONNX model.
        """
        # Tokenize
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=32,
            return_tensors="np"
        )
        input_ids = encoded["input_ids"].astype(np.int64)
        attention_mask = encoded["attention_mask"].astype(np.int64)

        # Run ONNX inference
        outputs = self.session.run(
            ["sentence_embedding"],   # output name we defined in export_onnx.py
            {"input_ids": input_ids, "attention_mask": attention_mask}
        )
        embedding = outputs[0][0]   # shape (embedding_dim,)
        return embedding

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict command for a given text string.
        Returns:
            (command_label, confidence)
            If confidence < threshold, command_label is "reject".
        """
        emb = self._get_embedding(text)

        # Compute cosine similarity with each prototype
        # Cosine = dot(emb, proto) / (||emb|| * ||proto||)
        norms = np.linalg.norm(emb) * np.linalg.norm(self.prototypes, axis=1)
        similarities = np.dot(self.prototypes, emb) / norms
        max_idx = np.argmax(similarities)
        confidence = similarities[max_idx]

        if confidence >= self.threshold:
            return self.labels[max_idx], confidence
        else:
            return "reject", confidence

# with some standalone test commands
if __name__ == "__main__":
    classifier = CommandClassifier(
        onnx_model_path="models/model_quantized.onnx",
        prototypes_path="models/prototypes.npy",
        labels_path="models/class_labels.json",
        threshold_path="models/threshold.json"
    )
    test_inputs = [
        "increase the volume",
        "um, pause music",
        "what is the weather",
        "lower brightness",
        "stop the vehicle"
    ]
    for inp in test_inputs:
        cmd, conf = classifier.predict(inp)
        print(f"Input: '{inp}' -> {cmd} (confidence={conf:.3f})")