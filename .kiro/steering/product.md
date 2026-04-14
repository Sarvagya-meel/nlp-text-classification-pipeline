---
inclusion: always
---

# NLP Text Classification Pipeline

## Purpose
A modular, end-to-end NLP pipeline for text classification (e.g. sentiment analysis, spam detection). Designed to be interview-ready and production-style — clean, explainable, and easy to extend.

## Core Capabilities
- Ingest raw text data from CSV files
- Preprocess and normalize text (tokenization, stopword removal, stemming/lemmatization)
- Extract features via TF-IDF with unigrams and bigrams
- Train and compare five classifiers: Naive Bayes, Logistic Regression, LinearSVC, Decision Tree, Random Forest
- Evaluate each model with accuracy, precision, recall, and F1
- Persist trained models and run inference on new input

## Out of Scope
- No RAG pipelines, LLM fine-tuning, or embedding-based approaches
- No distributed systems or complex infrastructure

## Quality Bar
- Pipeline runs end-to-end without errors
- All models evaluated under identical conditions for fair comparison
- Code is modular, single-responsibility, and easy to walk through in an interview setting
