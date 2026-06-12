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
```

Create a virtual environment, install dependencies,
generate dataset, train prototypes, export ONNX model, quantize to INT8, and run evaluation.


## To Run Model:
### To Run Model in Terminal
   - Just run,
```bash
source venv/bin/activate
python run_model.py
```

### To Run Model in Web Interface (Streamlit)
   - Just run,
```bash
source venv/bin/activate
python app.py
```
then Type a voice command text and see the prediction.

## To Run Evaluation
   - Just Run,
```bash
source venv/bin/activate
python src/evaluate.py
```
Produces classification report, OOS rejection rate, confusion matrix.

## Example Inputs & Outputs

| Input Text | Prediction |
|------------|------------|
| "increase the volume" | `increase_volume (0.98)` |
| "um, pause music" | `pause_music (0.92)` |
| "lower brightness" | `decrease_brightness (0.88)` |
| "what is the weather" | `reject (0.45)` |
| "stop the vehicle" | `stop_vehicle (0.91)` |

---

## Known Limitations

- Synthetic noise approximates real ASR errors, but actual ASR logs would improve robustness.
- Accent simulation is rule-based; real Indian English variations may be more diverse.
- Prototype-based classification cannot learn complex intra-class distinctions. Fine-tuning the embedding model could improve accuracy, but would reduce extensibility.
- Latency benchmarks were measured on an Apple M4 MacBook CPU; performance on edge devices may vary.

---

## Extensibility

The system is designed to support new commands without retraining the embedding model.

### Example: Add a New Command

Suppose you want to add a new command:

```text
open the sunroof
```

### Step 1: Add Training Examples

Add 5–10 example phrases to:

```text
data/raw_commands.json
```

Example:

```json
{
  "open_sunroof": [
    "open the sunroof",
    "please open the sunroof",
    "sunroof open",
    "can you open the sunroof",
    "open my sunroof"
  ]
}
```

### Step 2: Recompute Prototypes

Run:

```bash
python src/train.py
```

This recomputes:

- Class prototypes (`prototypes.npy`)
- Class labels (`class_labels.json`)
- OOS threshold (`threshold.json`)

Typical execution time:

```text
~2 seconds
```

### Step 3: Use the New Command

The classifier can now recognize:

```text
open the sunroof
```

### No Model Retraining Required

- No embedding model retraining

- No ONNX model re-export

- No quantization step required

- Extensible by updating prototypes only
