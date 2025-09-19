# src/train.py

import os
import joblib
import mlflow
import mlflow.sklearn

from src.build_dataset import load_imdb_dataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt
import seaborn as sns


# --- 🔹 Force MLflow to log locally instead of using mlflow-artifacts ----
mlruns_path = os.path.abspath("mlruns")
mlflow.set_tracking_uri(f"file:///{mlruns_path.replace(os.sep, '/')}")
mlflow.set_experiment("Sentiment-Analysis")


def train_and_evaluate(test_size=0.2, random_state=42, max_features=5000, ngram_range=(1, 2)):
    """
    Train a TF-IDF + Logistic Regression pipeline on IMDB dataset,
    evaluate it, log everything in MLflow, and save the model.
    """

    # --- 1. Load dataset -------------------------------------------------
    df = load_imdb_dataset()
    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # --- 2. Build pipeline ----------------------------------------------
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english"
        )),
        ("clf", LogisticRegression(
            solver="liblinear",
            max_iter=1000,
            random_state=random_state
        ))
    ])

    # --- 3. MLflow experiment logging -----------------------------------
    with mlflow.start_run(run_name=f"tfidf_{max_features}_features"):
        # log parameters
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("ngram_range", ngram_range)
        mlflow.log_param("classifier", "LogisticRegression")

        # --- 4. Train ----------------------------------------------------
        pipeline.fit(X_train, y_train)

        # --- 5. Evaluate -------------------------------------------------
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        print("Accuracy:", acc)
        print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))
        print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))

        # --- 6. Confusion matrix as artifact -----------------------------
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        cm_path = "outputs/confusion_matrix.png"
        os.makedirs("outputs", exist_ok=True)
        plt.savefig(cm_path)
        plt.close()

        mlflow.log_artifact(cm_path)

        # --- 7. Save model (to disk + MLflow) ----------------------------
        model_dir = "outputs/model"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "logreg_tfidf_pipeline.pkl")
        joblib.dump(pipeline, model_path)
        print(f"\nSaved trained pipeline to: {model_path}")

        # log model in MLflow
        mlflow.sklearn.log_model(pipeline, "sentiment_model")


if __name__ == "__main__":
    # 🔹 Run one or more experiments
    print("\n=== Run 1: 5000 features, unigrams+bigrams ===")
    train_and_evaluate(max_features=5000, ngram_range=(1, 2))

    print("\n=== Run 2: 10000 features, unigrams+bigrams ===")
    train_and_evaluate(max_features=10000, ngram_range=(1, 2))


# Write metrics to a file release_notes.md
with open("release_notes.md", "w") as f:
    f.write(f"# Sentiment Analysis Model\n\n")
    f.write(f"**Commit SHA:** {os.getenv('GITHUB_SHA', 'local-run')}\n\n")
    f.write("## Parameters\n")
    f.write(f"- max_features: {max_features}\n")
    f.write(f"- ngram_range: {ngram_range}\n\n")
    f.write("## Metrics\n")
    f.write(f"- Accuracy: {acc:.4f}\n")
    f.write(f"- Precision: {prec:.4f}\n")
    f.write(f"- Recall: {rec:.4f}\n")
    f.write(f"- F1 Score: {f1:.4f}\n")

