import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report


# =========================
# CONFIG (no CLI)
# =========================
DATA_PATH = os.path.join("data", "raw", "IMDB Dataset.csv")  # Kaggle file name usually exactly this
OUTPUT_DIR = "outputs"
RANDOM_SEED = 42
TEST_SIZE = 0.2

os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_text(text: str) -> str:
    """
    Minimal text cleaning for a medium-level baseline:
    - lowercasing
    - remove HTML tags
    - remove URLs
    - keep letters/numbers and basic punctuation spacing
    """
    text = str(text).lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)  # keep apostrophes for contractions
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            f"Put the Kaggle CSV here: data/raw/IMDB Dataset.csv"
        )
    df = pd.read_csv(path)
    # Expected columns: 'review', 'sentiment'
    expected = {"review", "sentiment"}
    if not expected.issubset(set(df.columns)):
        raise ValueError(f"Expected columns {expected}, got: {df.columns.tolist()}")
    return df


def save_confusion_matrix(y_true, y_pred, out_path: str, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=["negative", "positive"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["negative", "positive"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, values_format="d")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def top_features_from_logreg(pipe: Pipeline, top_n: int = 25) -> pd.DataFrame:
    """
    Extract top positive/negative features from logistic regression.
    Works for binary classification (negative/positive).
    """
    vec = pipe.named_steps["vectorizer"]
    clf = pipe.named_steps["clf"]

    feature_names = np.array(vec.get_feature_names_out())
    coefs = clf.coef_.ravel()

    top_pos_idx = np.argsort(coefs)[-top_n:][::-1]
    top_neg_idx = np.argsort(coefs)[:top_n]

    rows = []
    for i in top_pos_idx:
        rows.append({"feature": feature_names[i], "weight": float(coefs[i]), "class_direction": "positive"})
    for i in top_neg_idx:
        rows.append({"feature": feature_names[i], "weight": float(coefs[i]), "class_direction": "negative"})

    return pd.DataFrame(rows)


def evaluate_model(name: str, pipe: Pipeline, X_train, X_test, y_train, y_test) -> dict:
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, pos_label="positive")

    # Save confusion matrix image (for report screenshots)
    cm_path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{name}.png")
    save_confusion_matrix(y_test, preds, cm_path, f"Confusion Matrix - {name}")

    # Save top features
    feats = top_features_from_logreg(pipe, top_n=25)
    feats_path = os.path.join(OUTPUT_DIR, f"top_features_{name}.csv")
    feats.to_csv(feats_path, index=False)

    # Print a short report to console
    print("\n" + "=" * 70)
    print(f"MODEL: {name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 (positive): {f1:.4f}")
    print("Classification report:")
    print(classification_report(y_test, preds, digits=4))

    return {
        "model": name,
        "accuracy": float(acc),
        "f1_positive": float(f1),
        "confusion_matrix_path": cm_path,
        "top_features_path": feats_path,
        "notes": "Binary sentiment classification on IMDB reviews."
    }


def main():
    df = load_data(DATA_PATH)

    # Basic cleaning + duplicates
    df = df.drop_duplicates()
    df["review"] = df["review"].apply(clean_text)

    # Train/test split (strict rule: never train on test) 
    X_train, X_test, y_train, y_test = train_test_split(
        df["review"],
        df["sentiment"],
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df["sentiment"]
    )

    # -----------------------------
    # EXPERIMENTS (medium-level)
    # 1) BoW unigram + C=1.0
    # 2) TF-IDF bigram + C=2.0
    # (This satisfies "try at least 2 different parameter settings")
    # -----------------------------
    experiments = {
        "bow_unigram_C1": Pipeline([
            ("vectorizer", CountVectorizer(max_features=30000, ngram_range=(1, 1))),
            ("clf", LogisticRegression(max_iter=200, C=1.0, solver="liblinear"))
        ]),
        "tfidf_bigram_C2": Pipeline([
            ("vectorizer", TfidfVectorizer(max_features=50000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=300, C=2.0, solver="liblinear"))
        ]),
    }

    results = []
    for name, pipe in experiments.items():
        res = evaluate_model(name, pipe, X_train, X_test, y_train, y_test)
        results.append(res)

    # Save results JSON (for GitHub + report)
    out_json = os.path.join(OUTPUT_DIR, "results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Choose best model by F1
    best = max(results, key=lambda x: x["f1_positive"])
    print("\n" + "=" * 70)
    print("BEST MODEL (by F1):", best["model"])
    print("Saved outputs in:", OUTPUT_DIR)
    print("Results JSON:", out_json)


if __name__ == "__main__":
    main()
