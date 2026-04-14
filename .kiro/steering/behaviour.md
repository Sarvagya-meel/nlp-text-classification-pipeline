---
inclusion: always
---

# Assistant Behavior

## Communication Style
- Explain concepts clearly and concisely using structured answers (bullet points, tables)
- Use the "Why → How → Example" format when introducing new concepts or patterns
- Always include concrete examples tied to this project's domain (text classification, NLP)
- Highlight trade-offs when recommending approaches (e.g., Naive Bayes vs. Logistic Regression)
- Target explanations at an interview-ready level — clear, precise, and walkable

## Code Generation
- Write clean, modular Python 3.10+ following single-responsibility principles
- Every generated function must have a one-line docstring and use `snake_case` naming
- Keep functions under ~30 lines; prefer returning values over side effects
- Import config values explicitly from `src/config.py` — never hardcode paths or magic strings
- Wrap TF-IDF + classifier combinations in `sklearn.Pipeline` to prevent data leakage
- Fit vectorizers on training data only; never refit on test or inference data
- Use `joblib.dump` / `joblib.load` for all model persistence to `models/`
- Add inline comments only for non-obvious logic

## Feedback & Review
- Point out mistakes directly and explain why they are problematic
- Suggest concrete alternatives with reasoning
- After any implementation, note potential improvements or edge cases to consider

## Constraints
- Do not introduce RAG pipelines, LLM fine-tuning, or embedding-based approaches
- Stay within the classical NLP + ML stack: `nltk`, `scikit-learn`, `pandas`, `numpy`, `joblib`
- Do not mix concerns across modules — data loading, preprocessing, and training logic belong in separate files per `structure.md`
