import joblib

# load the trained pipeline
model = joblib.load("logreg_tfidf_pipeline_release.pkl")

#print(type(model))  # just to confirm what kind of object this is

texts = [
    "I really loved this movie, it was fantastic!",
    "Worst film ever, terrible acting and boring plot."
]

# Predict sentiment
predictions = model.predict(texts)
print(predictions)  # Expected: [1 0]

# Quick smoke test
for text in texts:
    pred = model.predict([text])[0]
    label = "Positive" if pred == 1 else "Negative"
    print(f"Text: {text} --> {label}")

