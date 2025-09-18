# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from build_dataset import load_imdb_dataset   # reuse function to load data

def extract_features():
    # 1. Load the dataset
    df = load_imdb_dataset()
    print("Loaded dataset:", df.shape)

    # 2. Separate input (X) and output (y)
    X_text = df["text"]      # raw text reviews
    y = df["label"]          # labels (0 = negative, 1 = positive)

    # 3. Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,   # keep only top 5000 words to reduce dimensionality
        ngram_range=(1,2),   # unigrams + bigrams (e.g. "good", "not good")
        stop_words="english" # extra stopword removal
    )

    # 4. Fit the vectorizer on text and transform it into numerical matrix
    X = vectorizer.fit_transform(X_text)

    print("Feature matrix shape:", X.shape)

    return X, y, vectorizer

if __name__ == "__main__":
    X, y, vectorizer = extract_features()
    print("Example feature names:", vectorizer.get_feature_names_out()[:20])
