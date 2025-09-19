# Sentiment Analysis Model (TF-IDF + Logistic Regression)

**Release:** v1 (example — replace with your actual release tag)  
**Commit SHA:** <commit sha from the release notes>  

## 📌 Overview
This is a binary sentiment classifier trained on the IMDB 50k movie reviews dataset.  
It predicts whether a review is **positive (1)** or **negative (0)**.  

## 🧑‍💻 Training
- Dataset: [Kaggle IMDB Dataset of 50k Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  
- Features: TF-IDF vectorization (max features = 5000, n-gram range = (1,2))  
- Model: Logistic Regression (liblinear solver, max_iter=1000)  

## 📊 Metrics
- Accuracy: ~0.88 (on test split)
- Precision: ~0.88
- Recall: ~0.88
- F1 Score: ~0.88  

*(Exact numbers depend on the run — copy them from your MLflow output.)*

## ✅ How to Use
Download `logreg_tfidf_pipeline.pkl` from the release, then:

```python
import joblib
model = joblib.load("logreg_tfidf_pipeline.pkl")
print(model.predict(["This was a great movie!"]))
