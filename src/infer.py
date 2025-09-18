# src/infer.py

import joblib   # to load the saved pipeline

def load_model(model_path="outputs/model/logreg_tfidf_pipeline.pkl"):
    """
    Load the trained pipeline (TF-IDF + Logistic Regression).
    """
    pipeline = joblib.load(model_path)
    return pipeline

def predict_sentiment(pipeline, texts):
    """
    Predict sentiment for a list of input texts.
    Returns class labels and probabilities.
    """
    preds = pipeline.predict(texts)           # 0=negative, 1=positive
    probs = pipeline.predict_proba(texts)     # probability for each class
    return preds, probs

if __name__ == "__main__":
    # 1. Load model
    pipeline = load_model()

    # 2. Example texts
    examples = [
        "I absolutely loved this movie, it was fantastic!",
        "This was the worst film I have ever seen.",
        "The plot was okay, but the acting was brilliant.",
        "Boring and too long, I nearly fell asleep."
    ]

    # 3. Run predictions
    preds, probs = predict_sentiment(pipeline, examples)

    # 4. Print results
    for text, label, prob in zip(examples, preds, probs):
        sentiment = "Positive" if label == 1 else "Negative"
        print(f"\nReview: {text}")
        print(f"Predicted: {sentiment} (Probabilities: {prob})")
