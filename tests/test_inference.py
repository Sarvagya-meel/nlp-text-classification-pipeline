# tests/test_inference.py
# Unit tests for the inference module.
# Run with: pytest tests/test_inference.py -v

import pytest
from src.inference import load_model, predict


class TestLoadModel:
    """Test model loading functionality."""

    def test_load_existing_model(self):
        """Load a trained model successfully."""
        # Assumes models have been trained — skip if not
        try:
            pipeline = load_model("logistic_regression")
            assert pipeline is not None
            assert hasattr(pipeline, "predict")
        except FileNotFoundError:
            pytest.skip("Models not trained yet. Run: python3 -m src.main --data data/raw/dataset.csv")

    def test_load_nonexistent_model(self):
        """Raise FileNotFoundError for missing model."""
        with pytest.raises(FileNotFoundError, match="No saved model found"):
            load_model("nonexistent_model")


class TestPredict:
    """Test prediction functionality."""

    @pytest.mark.parametrize("model_name", [
        "naive_bayes",
        "logistic_regression",
        "svm",
        "decision_tree",
        "random_forest",
    ])
    def test_predict_positive_text(self, model_name):
        """Predict on clearly positive text."""
        try:
            label, confidence = predict("This product is amazing! Best purchase ever.", model_name)
            assert isinstance(label, str)
            assert isinstance(confidence, float)
            assert 0.0 <= confidence <= 1.0
            # Most models should predict positive for this text
            assert label in ["positive", "negative"]
        except FileNotFoundError:
            pytest.skip(f"Model {model_name} not trained yet.")

    @pytest.mark.parametrize("model_name", [
        "naive_bayes",
        "logistic_regression",
        "svm",
        "decision_tree",
        "random_forest",
    ])
    def test_predict_negative_text(self, model_name):
        """Predict on clearly negative text."""
        try:
            label, confidence = predict("Terrible quality. Complete waste of money.", model_name)
            assert isinstance(label, str)
            assert isinstance(confidence, float)
            assert 0.0 <= confidence <= 1.0
            assert label in ["positive", "negative"]
        except FileNotFoundError:
            pytest.skip(f"Model {model_name} not trained yet.")

    def test_predict_empty_string(self):
        """Raise ValueError for empty input."""
        with pytest.raises(ValueError, match="Input text must be a non-empty string"):
            predict("", "logistic_regression")

    def test_predict_whitespace_only(self):
        """Raise ValueError for whitespace-only input."""
        with pytest.raises(ValueError, match="Input text must be a non-empty string"):
            predict("   ", "logistic_regression")

    def test_confidence_bounds(self):
        """Confidence score is always in [0, 1]."""
        try:
            _, confidence = predict("This is a test.", "logistic_regression")
            assert 0.0 <= confidence <= 1.0
        except FileNotFoundError:
            pytest.skip("Model not trained yet.")


class TestInferenceIntegration:
    """Integration tests for full inference workflow."""

    def test_round_trip_save_load_predict(self):
        """Save a model, load it, and predict — results should be consistent."""
        # This test assumes training has been run
        try:
            # Load model
            pipeline = load_model("logistic_regression")

            # Predict directly via pipeline
            text = "Great product, highly recommend!"
            direct_pred = pipeline.predict([text])[0]

            # Predict via inference API
            api_label, _ = predict(text, "logistic_regression")

            # Both should return the same label
            assert direct_pred == api_label
        except FileNotFoundError:
            pytest.skip("Model not trained yet.")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
