import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import CommandClassifier

def test_classifier():
    classifier = CommandClassifier(
        onnx_model_path="models/model_quantized.onnx",
        prototypes_path="models/prototypes.npy",
        labels_path="models/class_labels.json",
        threshold_path="models/threshold.json"
    )
    # known clean command used for testing (should be classified correctly with high confidence)
    cmd, conf = classifier.predict("increase the volume")
    assert cmd == "increase_volume" or conf > 0.7, "Failed on clean command"
    # OOS(out-of-scope) command (should be rejected)
    cmd, _ = classifier.predict("what is the weather")
    assert cmd == "reject", "Failed to reject OOS"
    print("All tests passed.")

if __name__ == "__main__":
    test_classifier()