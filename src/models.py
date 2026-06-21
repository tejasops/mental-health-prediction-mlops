"""Model definitions and training utilities."""
import pickle
import os

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    BaggingClassifier, AdaBoostClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


def get_models():
    """Return a dict of model_name -> model instance."""
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(max_depth=20, random_state=0),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=20, random_state=0
        ),
        'Extra Trees': ExtraTreesClassifier(
            n_estimators=100, max_depth=20, random_state=0
        ),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Bagging (KNN)': BaggingClassifier(
            estimator=KNeighborsClassifier(), n_estimators=10, random_state=0
        ),
        'AdaBoost': AdaBoostClassifier(
            n_estimators=100, random_state=0
        ),
        'MLP Neural Network': MLPClassifier(
            hidden_layer_sizes=(100,), max_iter=500, random_state=0
        ),
    }


def train_model(model, X_train, y_train):
    """Train and return the model."""
    model.fit(X_train, y_train)
    return model


def save_model(model, filepath=None):
    """Save a model to disk with pickle."""
    if filepath is None:
        filepath = os.path.join(
            os.path.dirname(__file__), '..', 'models', 'best_model.pkl'
        )
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def load_model(filepath=None):
    """Load a model from disk."""
    if filepath is None:
        filepath = os.path.join(
            os.path.dirname(__file__), '..', 'models', 'best_model.pkl'
        )
    with open(filepath, 'rb') as f:
        return pickle.load(f)
