"""
Interactive demo for the classifier.
"""

from src.inference import CommandClassifier

def main():
    classifier = CommandClassifier(
        onnx_model_path="models/model_quantized.onnx",
        prototypes_path="models/prototypes.npy",
        labels_path="models/class_labels.json",
        threshold_path="models/threshold.json"
    )
    print("Command Classifier Demo (type 'exit' to quit)")
    while True:
        user_input = input("\nEnter a voice command text: ").strip()
        if user_input.lower() == "exit":
            break
        cmd, conf = classifier.predict(user_input)
        print(f"Prediction: {cmd} (confidence = {conf:.3f})")

if __name__ == "__main__":
    main()