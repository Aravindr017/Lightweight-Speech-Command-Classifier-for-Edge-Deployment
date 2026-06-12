import streamlit as st
import sys
# import os
sys.path.append("src")      # add the src folder to path so import files eaisly
from inference import CommandClassifier     # import the classifier class from inference.py

st.set_page_config(page_title="Voice Command Classifier", layout="centered")        # configure browser page
st.title("Edge Command Classifier (Offline)")
st.markdown("Enter a voice command text (ASR output). The model will classify it or reject if out-of-scope.")

@st.cache_resource      # used to cache the loaded model so it doesn't reload on every input change
def load_classifier():
    return CommandClassifier(
        onnx_model_path="models/model_quantized.onnx",
        prototypes_path="models/prototypes.npy",
        labels_path="models/class_labels.json",
        threshold_path="models/threshold.json"
    )

classifier = load_classifier()

user_input = st.text_input("Your command:", placeholder="e.g., increase the volume")
if user_input:
    with st.spinner("Classifying..."):
        pred, conf = classifier.predict(user_input)
    st.success(f"**Prediction:** {pred}")
    st.metric("Confidence", f"{conf:.3f}")      # it is value b/w 0 and 1 , higher means more confident (cosine similarity score)
    if pred == "reject":        # OOS check case (rejected)
        st.warning("This input was not recognised as a known command.")