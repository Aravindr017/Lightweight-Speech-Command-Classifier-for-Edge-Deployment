# Lightweight Speech Command Classifier for Edge Deployment

## Overview
This project implements a **semantic command classifier** for automotive and IoT voice systems.  
It takes a text string (from an ASR engine) and maps it to one of 14 predefined commands, or rejects it if the input is out-of-scope (OOS).  
The model is designed to run **fully offline** on edge devices (mobile, infotainment) with low latency and small memory footprint.

**Key features**  
- 10 core commands + 4 extension commands (brightness, start/stop vehicle).  
- Out‑of‑scope rejection using a cosine similarity threshold.  
- INT8 quantized ONNX model (~16 MB) – meets ≤25 MB constraint.  
- Inference latency <100 ms on CPU.  
- Extensible – new commands can be added without retraining the embedding model.  
- Robust to ASR noise (synthetic word drops, typos, fillers, Indian accent simulation).

## How It Works

1. **Training phase (offline)**  
   - Generate synthetic dataset with noise and OOS examples.  
   - Compute sentence embeddings using a pre‑trained `paraphrase-MiniLM-L3-v2` model.  
   - For each command, average its training embeddings → **prototype vector**.  
   - Choose a threshold that separates in‑scope and OOS validation samples (maximises F1).

2. **Inference phase (on device)**  
   - Input text → tokenize → ONNX model returns a sentence embedding.  
   - Compute cosine similarity with all command prototypes.  
   - If max similarity ≥ threshold → return command label; else `"reject"`.

## Installation & Running

### Prerequisites
- Python 3.11+
- Internet connection for first run (to download models & tokenizer). Afterwards, fully offline.

### Setup
```bash
git clone https://github.com/yourusername/command-classifier.git
cd command-classifier
chmod +x setup.sh
./setup.sh
