# Utility functions to ensure tokenizer is downloaded
# after one time downloaded it is cached locally (then able to run offline)

from transformers import AutoTokenizer

def get_tokenizer(model_name="microsoft/MiniLM-L3-H384-uncased"):
    return AutoTokenizer.from_pretrained(model_name)

if __name__ == "__main__":
    # Pre‑download tokenizer for offline use
    tokenizer = get_tokenizer()
    print("Tokenizer downloaded and cached.")