
"""
where creates dataset.csv with noise (nlpaug and manual)
Creates a CSV file containing text commands with synthetic ASR noise and OOS sentences.
The dataset is split into train/val/test.
"""

import json
import random
import pandas as pd
import nlpaug.augmenter.word as naw
from typing import List

# configuration
EXAMPLES_PER_COMMAND = 20          # number of noisy variants per command for training
CLEAN_VAL_TEST_PER_COMMAND = 3     # number of clean examples per command for val/test
OOS_SENTENCES = [       # some oos sentences that are not commands (for validation and testing)
    "what is the weather today",
    "navigate to nearest petrol station",
    "call John",
    "hello",
    "turn off the air conditioning",
    "how are you",
    "set a timer",
    "open the trunk",
    "what's the stock price",
    "thank you",
    "okay Google",
    "play video",
]

# Noise Augmentation functions (adding noise to simulate ASR errors)
# Random word deletion (returns a list)
word_drop_aug = naw.RandomWordAug(action="delete", aug_p=0.3)

# 2. Typo simulation
def add_typos(text: str) -> str:
    # Replace some words with common misspellings
    typo_map = [("the", "teh"), ("volume", "volum"), ("music", "musik"),
                ("next", "nekt"), ("previous", "previuos"), ("brightness", "brightnes")]
    for original, wrong in typo_map:
        if original in text and random.random() < 0.3:
            text = text.replace(original, wrong)
    return text

# Filler word insertion (because client is human)
FILLERS = ["um", "uh", "like", "actually"]
def insert_filler(text: str) -> str:
    if random.random() < 0.2:
        words = text.split()
        position = random.randint(0, len(words))
        filler = random.choice(FILLERS)
        words.insert(position, filler)
        return " ".join(words)
    return text

# Indian accent simulation (pronounciation changes)
def apply_indian_accent(text: str) -> str:
    #Mimics common Indian English pronunciation changes.
    accent_rules = [("ve", "we"), ("vo", "wo"), ("th", "d"), ("v", "w"), ("c", "k")]
    for src, tgt in accent_rules:
        text = text.replace(src, tgt)
    return text

def augment_text(text: str) -> str:
    #Apply a random combination of noise augmentations. Always returns a string.
    # Word drop returns a list; take the first element
    if random.random() < 0.4:
        augmented_list = word_drop_aug.augment(text)
        if augmented_list and isinstance(augmented_list, list):
            text = augmented_list[0]
        elif isinstance(augmented_list, str):
            text = augmented_list
    # Apply other augmentations (they work on strings)
    if random.random() < 0.3:
        text = add_typos(text)
    if random.random() < 0.2:
        text = insert_filler(text)
    if random.random() < 0.3:
        text = apply_indian_accent(text)
    return text

# Main function to generate the dataset
def generate_dataset() -> pd.DataFrame:
    # Load base command phrases from JSON
    with open("data/raw_commands.json", "r") as f:
        commands_dict = json.load(f)

    rows = []

    # Training samples (augmented)
    for label, phrases in commands_dict.items():
        for _ in range(EXAMPLES_PER_COMMAND):
            base_phrase = random.choice(phrases)
            noisy_phrase = augment_text(base_phrase)
            rows.append({"text": noisy_phrase, "label": label, "split": "train"})

    # Validation and test samples (clean)
    for label, phrases in commands_dict.items():
        for split_name in ["val", "test"]:
            for _ in range(CLEAN_VAL_TEST_PER_COMMAND):
                clean_phrase = random.choice(phrases)
                rows.append({"text": clean_phrase, "label": label, "split": split_name})

    # Out‑of‑scope (OOS) samples
    for oos_text in OOS_SENTENCES:
        rows.append({"text": oos_text, "label": "oos", "split": "val"})
        rows.append({"text": oos_text, "label": "oos", "split": "test"})

    # Create DataFrame and shuffle
    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("data/dataset.csv", index=False)
    print(f"Dataset saved to data/dataset.csv")
    print(f"Total samples: {len(df)}")
    print(f"Train size: {len(df[df['split']=='train'])}")
    print(f"Val size:   {len(df[df['split']=='val'])}")
    print(f"Test size:  {len(df[df['split']=='test'])}")