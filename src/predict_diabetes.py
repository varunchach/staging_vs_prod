import joblib
import numpy as np
from sklearn.datasets import load_diabetes


PRODUCTION_MODEL_PATH = "models/diabetes_production_model.pkl"


def load_production_model():
    """Load the model that has been promoted to production."""
    model = joblib.load(PRODUCTION_MODEL_PATH)
    return model


def predict_single(sample_features):
    """
    sample_features: list or array of length 10 (diabetes dataset has 10 features)
    Returns the predicted disease progression score.
    """
    model = load_production_model()
    sample_array = np.array(sample_features).reshape(1, -1)
    prediction = model.predict(sample_array)[0]
    return prediction


if __name__ == "__main__":
    # For demo, we’ll just take the first record from the diabetes dataset.
    diabetes = load_diabetes()
    first_sample = diabetes.data[0]

    print("Using production model to predict on one sample...")
    pred = predict_single(first_sample)
    print(f"Predicted diabetes progression score: {pred:.2f}")
