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

## Entry 8 — L1 vs L2 Regularization

**Date:** 2026-04-19

### Concept Learned
Regularization adds a penalty to the loss function to prevent overfitting by discouraging
large model weights. L1 (Lasso) pushes some weights exactly to zero — automatic feature
selection. L2 (Ridge) shrinks all weights toward zero but never to exactly zero — more
stable when features are correlated (common in TF-IDF). The `C` parameter controls
strength: small C = strong regularization = simpler model.

### Example
```python
from sklearn.linear_model import LogisticRegression

# L2 (default) — shrinks all weights, stable on correlated TF-IDF features
lr_l2 = LogisticRegression(l1_ratio=0.0, C=1, solver="saga")

# L1 — zeroes out irrelevant features, produces sparse model
lr_l1 = LogisticRegression(l1_ratio=1.0, C=10, solver="saga")

# Observed results on this dataset:
# L2 at C=0.1 → test_score=0.90  (converges quickly)
# L1 at C=0.1 → test_score=0.50  (too aggressive, needs higher C)
# L2 at C=1   → test_score=0.95  (sweet spot)
# L1 at C=10  → test_score=0.85  (needs more freedom to perform)
```

### Interview Question
**Q: When would you choose L1 over L2 regularization for a text classifier?**
A: Choose L1 when you suspect many TF-IDF features are irrelevant and want automatic
feature selection — L1 zeroes out those weights, producing a sparse, interpretable model.
Choose L2 when features are correlated (as they often are in TF-IDF, where similar words
co-occur) — L2 distributes weight across correlated features rather than arbitrarily
zeroing some out. In practice, L2 is the safer default for NLP tasks.

---

## Entry 9 — Cross-Validation

**Date:** 2026-04-19

### Concept Learned
A single train/test split gives one score that depends on which samples happened to land
in the test set — it can be misleadingly high or low. K-fold cross-validation splits the
data into k folds, trains on k-1 folds and tests on the remaining fold, repeating k times.
The final score is mean ± std across all folds. StratifiedKFold preserves class distribution
in each fold — essential for classification tasks.

### Example
```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=skf, scoring="accuracy")

print(f"Mean: {scores.mean():.4f}  Std: {scores.std():.4f}")
# Mean: 0.9600  Std: 0.0374  → stable model

# Observed results:
# naive_bayes         mean=0.96  std=0.037  → stable
# logistic_regression mean=0.94  std=0.020  → very stable
# decision_tree       mean=0.81  std=0.037  → lower mean, confirms overfitting
# random_forest       mean=0.93  std=0.051  → slightly unstable (ensemble variance)
```

### Interview Question
**Q: Why is cross-validation more reliable than a single train/test split?**
A: A single split gives one score that depends on which 20% of data happened to be in
the test set — this can be lucky or unlucky. Cross-validation averages over k different
splits, giving a more stable estimate of real-world performance. The standard deviation
across folds also reveals model stability: high std means the model is sensitive to which
data it sees, which is a sign of overfitting or insufficient data.

---

## Entry 10 — Hyperparameter Tuning with GridSearchCV

**Date:** 2026-04-19

### Concept Learned
Hyperparameters are settings chosen before training (not learned from data) — e.g. TF-IDF
`max_features`, `ngram_range`, and LogReg `C`. GridSearchCV exhaustively tries all
combinations and selects the best using cross-validation internally, so the test set is
never touched during selection. This prevents data leakage from hyperparameter choice.

### Example
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "tfidf__max_features": [5000, 10000],
    "tfidf__ngram_range":  [(1, 1), (1, 2)],
    "clf__C":              [0.1, 1, 10],
    "clf__l1_ratio":       [0.0, 1.0],   # 0.0=L2, 1.0=L1
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy")
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
# {'clf__C': 10, 'clf__l1_ratio': 0.0, 'tfidf__max_features': 5000,
#  'tfidf__ngram_range': (1, 1)}
# Best CV score: 0.975  |  Test score: 0.950
```

### Interview Question
**Q: Why must hyperparameter tuning use cross-validation rather than the test set?**
A: If you tune hyperparameters by evaluating on the test set, you're effectively training
on the test set — the chosen hyperparameters are optimised for that specific test data,
not for unseen data. This is data leakage. GridSearchCV uses cross-validation on the
training set only, so the test set remains a true held-out evaluation of the final model.

---
