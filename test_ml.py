import os

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model

project_path = os.path.dirname(os.path.abspath(__file__))
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


@pytest.fixture(scope="module")
def split_data():
    """Load census.csv and split it the same way as train_model.py."""
    data = pd.read_csv(os.path.join(project_path, "data", "census.csv"))
    train, test = train_test_split(
        data, test_size=0.20, random_state=42, stratify=data["salary"]
    )
    return data, train, test


def test_train_test_split_size(split_data):
    """Test that the data is split 80/20 and processed into a numpy array."""
    data, train, test = split_data
    assert len(train) + len(test) == len(data)
    assert len(test) == pytest.approx(0.20 * len(data), abs=1)
    X_train, y_train, _, _ = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    assert isinstance(X_train, np.ndarray)
    assert X_train.shape[0] == y_train.shape[0] == len(train)


def test_train_model_algorithm(split_data):
    """Test that the model is a random forest and predicts 0 or 1 per row."""
    _, train, _ = split_data
    sample = train.sample(1000, random_state=42)
    X, y, _, _ = process_data(
        sample, categorical_features=cat_features, label="salary", training=True
    )
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(sample),)
    assert set(np.unique(preds)) <= {0, 1}


def test_compute_model_metrics():
    """Test that the metrics are 0.5 for a small example."""
    y = np.array([1, 1, 0, 0])
    preds = np.array([1, 0, 1, 0])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(0.5)
    assert fbeta == pytest.approx(0.5)
