# Project Structure

## Root Layout

nlp-text-classification-pipeline/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── utils.py
├── models/
├── reports/
│   ├── metrics/
│   └── figures/
├── notes/
│   ├── concepts.md
│   ├── formulas.md
│   ├── interview_qs.md
│   └── learning_log.md
├── tests/
├── requirements.txt
└── README.md

## Folder Responsibilities

### data/
- raw/: original datasets
- processed/: cleaned datasets

### src/
- Contains all core logic and pipeline code

### models/
- Stores serialized models (joblib/pickle)

### reports/
- metrics/: evaluation outputs
- figures/: plots and visualizations

### notes/
- concepts.md: definitions and examples
- formulas.md: key formulas
- interview_qs.md: interview questions
- learning_log.md: iterative learning notes

### tests/
- Unit tests (optional for now)

## File Responsibilities

- data_loader.py → load datasets
- preprocessing.py → clean and normalize text
- features.py → vectorization (TF-IDF)
- train.py → model training
- evaluate.py → metrics calculation
- inference.py → prediction logic
- utils.py → helper functions

## Rules
- Do not mix responsibilities across files
- Keep each file focused on one concern
- Maintain consistent naming across modules