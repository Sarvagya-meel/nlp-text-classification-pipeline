# Tech Stack

## Language
- Python 3.9+

## Core Libraries

| Purpose | Library |
|---|---|
| ML / Classifiers | scikit-learn |
| NLP Preprocessing | nltk |
| Data handling | pandas, numpy |
| Serialization | joblib |
| Evaluation | scikit-learn metrics |
| CLI | argparse (stdlib) |

## Constraints
- No deep learning frameworks (no PyTorch, TensorFlow, Keras)
- No LLM APIs (no OpenAI, HuggingFace Transformers, LangChain)
- No vector databases or RAG components
- Prefer scikit-learn pipelines (`Pipeline`, `FeatureUnion`) where it improves clarity
- Use `joblib` for saving/loading trained models and vectorizers

## Python Conventions
- Type hints on all function signatures
- Docstrings on all public functions and classes (Google style)
- No global mutable state — pass config/params explicitly
- Use `if __name__ == "__main__"` guards on all runnable scripts
