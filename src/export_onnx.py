"""
Exports the sentence embedding model to ONNX (FP32) and applies dynamic INT8 quantization.
Uses the legacy ONNX exporter (dynamo=False) and opset 18.
"""

import torch
import torch.nn as nn
import os
from transformers import AutoModel, AutoTokenizer
from onnxruntime.quantization import quantize_dynamic, QuantType

class EmbeddingModel(nn.Module):
    """Mean-pooling wrapper for the transformer."""
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids, attention_mask=attention_mask)
        # Mean pooling: sum of hidden states / number of non‑padding tokens
        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled = torch.sum(outputs.last_hidden_state * mask_expanded, dim=1) / torch.sum(mask_expanded, dim=1)
        return pooled

def export_fp32_onnx(output_path_fp32: str = "models/model_fp32.onnx"):
    """Export the model to FP32 ONNX using the legacy exporter (dynamo=False)."""
    model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    transformer = AutoModel.from_pretrained(model_name)
    transformer.eval()

    model = EmbeddingModel(transformer)
    model.eval()

    # Dummy input
    dummy_text = "increase the volume"
    encoded = tokenizer(
        dummy_text,
        return_tensors="pt",
        padding="max_length",
        max_length=32,
        truncation=True
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Export with legacy exporter (dynamo=False) and opset 18
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        output_path_fp32,
        input_names=["input_ids", "attention_mask"],
        output_names=["sentence_embedding"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "sentence_embedding": {0: "batch_size"}
        },
        opset_version=18,          # use 18 to avoid LayerNormalization conversion issues
        do_constant_folding=True,
        verbose=False,
        dynamo=False          
    )
    size_mb = os.path.getsize(output_path_fp32) / (1024 * 1024)
    print(f"FP32 ONNX model saved to {output_path_fp32}")
    print(f"Size: {size_mb:.2f} MB")

def quantize_onnx(input_fp32: str, output_int8: str = "models/model_quantized.onnx"):
    """Apply dynamic INT8 quantization (weights only)."""
    quantize_dynamic(
        model_input=input_fp32,
        model_output=output_int8,
        per_channel=False,
        reduce_range=False,
        weight_type=QuantType.QInt8,
    )
    size_mb = os.path.getsize(output_int8) / (1024 * 1024)
    print(f"INT8 quantized model saved to {output_int8}")
    print(f"Quantized size: {size_mb:.2f} MB")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    export_fp32_onnx("models/model_fp32.onnx")
    quantize_onnx("models/model_fp32.onnx", "models/model_quantized.onnx")
    print("Export and quantization complete.")