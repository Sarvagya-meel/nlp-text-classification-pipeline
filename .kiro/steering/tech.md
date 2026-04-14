# Technical Guidelines

## Language
- Python 3.10+

## Libraries
- pandas
- numpy
- scikit-learn
- nltk
- joblib
- matplotlib
- seaborn

## Optional Libraries
- spaCy (for advanced NLP)
- FastAPI (for API layer)

## Modeling Approach
- Use TF-IDF as baseline feature extraction
- Include n-grams (1,2) where useful
- Compare multiple classifiers:
  - Multinomial Naive Bayes
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Decision Tree
  - Random Forest

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score

## Engineering Practices
- Use modular functions across files
- Keep functions small and readable
- Prefer sklearn Pipelines where appropriate
- Avoid hardcoding paths
- Use config file for constants

## Code Style
- Clear variable naming
- Minimal but meaningful comments
- Follow consistent structure across modules

## Performance Considerations
- Avoid unnecessary recomputation
- Reuse vectorizers where possible
- Keep models lightweight for fast iteration