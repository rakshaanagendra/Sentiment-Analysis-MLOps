import argparse
import joblib

def load_model(model_path):
    """Load the trained model pipeline from a pickle file."""
    return joblib.load(model_path)

def predict(model, texts):
    """Run predictions on a list of input texts."""
    preds = model.predict(texts)
    probs = model.predict_proba(texts)
    for text, pred, prob in zip(texts, preds, probs):
        label = "Positive" if pred == 1 else "Negative"
        confidence = max(prob) * 100
        print(f"\nText: {text}\nPrediction: {label} ({confidence:.2f}% confidence)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentiment prediction using trained model")
    parser.add_argument("--model", type=str, default="outputs/model/logreg_tfidf_pipeline.pkl",
                        help="Path to the trained model pickle file")
    parser.add_argument("--text", type=str, nargs="+",
                        help="One or more texts to classify")

    args = parser.parse_args()

    # Load model
    model = load_model(args.model)

    # Predict
    predict(model, args.text)
