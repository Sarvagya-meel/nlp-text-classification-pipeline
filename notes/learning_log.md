# Learning Log — NLP Text Classification Pipeline

A running record of concepts learned, examples, and interview questions as the project progresses.
Each entry is added after a milestone is completed.

---

## Entry 1 — Project Setup & Pipeline Architecture

**Date:** 2026-04-15

### Concept Learned
An NLP classification pipeline has five distinct stages that must stay modular:
data loading → preprocessing → feature extraction → model training → evaluation.
Each stage lives in its own file with a single responsibility.

### Example
```
CSV file
  → data_loader.py   (load + validate)
  → preprocessing.py (clean + tokenize + lemmatize)
  → features.py      (TF-IDF vectorization)
  → train.py         (fit sklearn Pipeline per model)
  → evaluate.py      (metrics + plots)
```

### Interview Question
**Q: Why do we wrap TF-IDF and a classifier together in an sklearn Pipeline?**
A: To prevent data leakage. If we fit the vectorizer on the full dataset before splitting,
test data influences the vocabulary and IDF weights — inflating evaluation scores.
A Pipeline ensures the vectorizer is fitted only on training data during `pipeline.fit(X_train, y_train)`.

---

## Entry 2 — Text Preprocessing

**Date:** 2026-04-15

### Concept Learned
Preprocessing converts noisy raw text into clean, consistent tokens that TF-IDF
can turn into meaningful features. The order matters: clean first, then tokenize,
then remove stopwords, then lemmatize.

### Example
```
Raw:       "I absolutely LOVED this product!!! <br> Best buy ever."
Cleaned:   "i absolutely loved this product best buy ever"
Tokens:    ["i", "absolutely", "loved", "this", "product", "best", "buy", "ever"]
No stops:  ["absolutely", "loved", "product", "best", "buy", "ever"]
Lemmatized:["absolutely", "love", "product", "best", "buy", "ever"]
```

### Interview Question
**Q: What is the difference between stemming and lemmatization? When would you use each?**
A: Stemming chops word endings aggressively ("running" → "run", "happiness" → "happi") —
fast but produces non-real-word roots. Lemmatization maps words to their dictionary base form
("running" → "run", "better" → "good") — slower but linguistically correct.
Use stemming when speed matters more than quality; use lemmatization for production NLP pipelines.

---

## Entry 3 — TF-IDF & N-grams

**Date:** 2026-04-15

### Concept Learned
TF-IDF scores a word by how often it appears in a document (TF) multiplied by how rare
it is across all documents (IDF). Common words like "the" get low scores; distinctive
words get high scores. N-grams extend this to word pairs (bigrams) to capture context.

### Example
```python
from sklearn.feature_extraction.text import TfidfVectorizer

docs = ["not good product", "very good product", "terrible product"]
vec = TfidfVectorizer(ngram_range=(1, 2))
X = vec.fit_transform(docs)
# Vocabulary includes: "not good", "very good" — bigrams preserve sentiment context
# "not good" and "very good" are now distinct features, not just "good"
```

### Interview Question
**Q: Why use ngram_range=(1,2) instead of just unigrams?**
A: Unigrams treat each word independently — "not good" looks similar to "very good"
because both contain "good". Bigrams capture the phrase "not good" as a single feature,
preserving negation and context. The trade-off is a larger vocabulary, controlled by
`max_features`.

---

## Entry 4 — Model Training & Comparison

**Date:** 2026-04-15

### Concept Learned
Training all five classifiers on the same TF-IDF features and the same train/test split
makes their evaluation scores directly comparable. Each model has different strengths:
Naive Bayes is fast and simple; Logistic Regression is a strong linear baseline;
SVM handles high-dimensional text well; Decision Tree is interpretable but overfits;
Random Forest is robust via ensembling.

### Example
```python
pipelines = {
    "naive_bayes":         Pipeline([("tfidf", TfidfVectorizer()), ("clf", MultinomialNB())]),
    "logistic_regression": Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())]),
    "svm":                 Pipeline([("tfidf", TfidfVectorizer()), ("clf", LinearSVC())]),
    "decision_tree":       Pipeline([("tfidf", TfidfVectorizer()), ("clf", DecisionTreeClassifier())]),
    "random_forest":       Pipeline([("tfidf", TfidfVectorizer()), ("clf", RandomForestClassifier())]),
}
# All trained on same X_train, y_train — fair comparison
```

### Interview Question
**Q: Why does Naive Bayes work well for text classification despite its "naive" assumption?**
A: Naive Bayes assumes all features (words) are conditionally independent given the class.
This is obviously false in language, but in practice the assumption rarely hurts classification
because the model only needs to rank classes, not estimate exact probabilities.
It also trains very fast and works well on small datasets.

---

## Entry 5 — Evaluation Metrics

**Date:** 2026-04-15

### Concept Learned
Accuracy alone is misleading on imbalanced datasets. Precision, Recall, and F1 give
a fuller picture. Precision measures how reliable positive predictions are; Recall
measures how many actual positives were found; F1 balances both.

### Example
```
Spam classifier — 100 emails, 10 are spam:

Model predicts everything as "not spam":
  Accuracy  = 90/100 = 0.90  ← looks great, but useless
  Recall    = 0/10   = 0.00  ← caught zero spam
  F1        = 0.00           ← reveals the model is broken

Good model:
  Accuracy  = 0.95
  Precision = 0.89  (of predicted spam, 89% were actually spam)
  Recall    = 0.80  (caught 80% of all spam)
  F1        = 0.84  (balanced score)
```

### Interview Question
**Q: When would you prioritise Recall over Precision?**
A: When false negatives are more costly than false positives. Example: cancer screening —
missing a positive case (false negative) is far worse than a false alarm (false positive).
In spam filtering, the opposite is true — you'd rather let spam through than block
a legitimate email, so Precision matters more.

---

## Entry 6 — Overfitting & Underfitting

**Date:** 2026-04-15

### Concept Learned
Overfitting: model memorises training data, fails on new data (high train score, low test score).
Underfitting: model is too simple to learn the task (both scores low).
The gap between train and test score is the primary diagnostic signal.

### Example
```
Decision Tree (no depth limit):
  Train score: 1.00  ← memorised every training example
  Test score:  0.72  ← gap of 0.28 → OVERFIT

Naive Bayes on very small dataset:
  Train score: 0.62
  Test score:  0.60  ← both low → UNDERFIT

Logistic Regression (regularised):
  Train score: 0.91
  Test score:  0.87  ← gap of 0.04 → GOOD FIT
```

### Interview Question
**Q: How do you fix overfitting in a text classification model?**
A: Several approaches depending on the model:
- Add regularisation (lower C in LogReg/SVM reduces model complexity)
- Reduce TF-IDF vocabulary size (lower max_features)
- Limit tree depth (max_depth in DecisionTree, min_samples_leaf in RandomForest)
- Use cross-validation instead of a single split to get a more reliable estimate
- Collect more training data so the model can't memorise individual examples

---

## Entry 7 — Model Persistence & Inference

**Date:** 2026-04-15

### Concept Learned
Trained models are serialized to disk with joblib so they can be loaded and used
for inference without retraining. The full sklearn Pipeline (TF-IDF + classifier)
is saved as a single object — ensuring the same preprocessing is applied at inference.

### Example
```python
import joblib

# Save
joblib.dump(fitted_pipeline, "models/logistic_regression.joblib")

# Load and predict
pipeline = joblib.load("models/logistic_regression.joblib")
label = pipeline.predict(["This product is amazing!"])
# → ["positive"]
```

### Interview Question
**Q: Why save the entire Pipeline object rather than just the classifier weights?**
A: The Pipeline includes the fitted TfidfVectorizer with its learned vocabulary and
IDF weights. If you only save the classifier, you'd need to refit the vectorizer at
inference time — which would produce a different vocabulary and break predictions.
Saving the full Pipeline guarantees identical preprocessing at training and inference time.

---
