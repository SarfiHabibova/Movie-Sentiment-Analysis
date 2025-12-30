# Movie Review Sentiment Analysis (IMDB)

This project performs **sentiment analysis on IMDB movie reviews** using classical machine learning
and text vectorization techniques.  
The goal is to classify reviews as **positive** or **negative** based on their textual content.

---

## 1. Problem Statement
The task is to predict the **sentiment polarity** of a movie review using its text.

This is formulated as a **binary text classification problem**:
- `positive` sentiment
- `negative` sentiment

---

## 2. Data Description
- **Dataset:** IMDB Dataset
- **File path:** `data/raw/IMDB Dataset.csv`
- **Source:** Kaggle
- **Number of samples:** 50,000 movie reviews
- **Target column:** `sentiment`
- **Text column:** `review`

Each record contains a full movie review and its corresponding sentiment label.

---

## 3. Project Structure
AI-MOVIE-PROJECT/
│
├── data/
│ └── raw/
│ └── IMDB Dataset.csv
│
├── outputs/
│ ├── confusion_matrix_bow_unigram_C1.png
│ ├── confusion_matrix_tfidf_bigram_C2.png
│ ├── top_features_bow_unigram_C1.csv
│ ├── top_features_tfidf_bigram_C2.csv
│ └── results.json
│
├── main.py
├── README.md
└── requirements.txt

---

## 4. Methodology

### Text Cleaning
A custom text cleaning function is applied:
- Lowercasing
- Removing HTML tags
- Removing URLs
- Keeping letters, numbers, and basic punctuation
- Normalizing whitespace

---

### Feature Extraction
Two different vectorization strategies are implemented:

#### C1 — Bag of Words (Unigram)
- Count-based vectorization
- Uses single words (unigrams)
- Captures word frequency information

#### C2 — TF-IDF (Bigram)
- TF-IDF weighting scheme
- Uses word pairs (bigrams)
- Reduces impact of common but less informative words

---

### Model
- **Classifier:** Logistic Regression
- **Train-test split:** 80% training, 20% testing
- **Random seed:** 42
- Same model trained separately for C1 and C2 representations

---

## 5. Results

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### Output Files
All results are saved automatically in the `outputs/` directory:
- Confusion matrices for both approaches
- Top influential features for each model
- Final metrics summary in `results.json`

Example files:
- `confusion_matrix_bow_unigram_C1.png`
- `confusion_matrix_tfidf_bigram_C2.png`
- `top_features_bow_unigram_C1.csv`
- `top_features_tfidf_bigram_C2.csv`

---

## 6. Conclusion
Both models successfully learned sentiment patterns from movie reviews.

- **C2 (TF-IDF Bigram)** achieved better performance due to capturing contextual word relationships.
- **C1 (Bag of Words Unigram)** performed well as a simpler baseline model.

This comparison demonstrates the importance of feature representation in text classification tasks.

