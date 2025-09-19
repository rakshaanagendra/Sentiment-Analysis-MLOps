import os
import pandas as pd
from preprocess import preprocess_text   # reuse your preprocessing function

def load_imdb_dataset():
    file_path = "data/raw/acImdb/IMDB_Dataset/IMDB-Dataset.csv"
    df = pd.read_csv(file_path)
    df.rename(columns={"review": "text", "sentiment": "label"}, inplace=True)

    # Convert labels: positive → 1, negative → 0
    df["label"] = df["label"].map({"positive": 1, "negative": 0})

    return df


    # --- TRAIN DATA ---
    for label_type in ["pos", "neg"]:
        folder_path = os.path.join(base_path, "train", label_type)
        for file_name in os.listdir(folder_path):
            with open(os.path.join(folder_path, file_name), "r", encoding="utf-8") as f:
                text = f.read()
                # Apply preprocessing
                words = preprocess_text(text)
                data.append({
                    "text": " ".join(words),   # join tokens back into a string
                    "label": 1 if label_type == "pos" else 0   # 1=positive, 0=negative
                })

    # --- TEST DATA ---
    for label_type in ["pos", "neg"]:
        folder_path = os.path.join(base_path, "test", label_type)
        for file_name in os.listdir(folder_path):
            with open(os.path.join(folder_path, file_name), "r", encoding="utf-8") as f:
                text = f.read()
                words = preprocess_text(text)
                data.append({
                    "text": " ".join(words),
                    "label": 1 if label_type == "pos" else 0
                })

    return pd.DataFrame(data)

if __name__ == "__main__":
    df = load_imdb_dataset()
    print("Dataset shape:", df.shape)
    print(df.head())