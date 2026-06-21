"""Model evaluation utilities."""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from sklearn.model_selection import cross_val_score


def evaluate_model(model, X_test, y_test):
    """Evaluate a model and return a metrics dict."""
    y_pred = model.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'classification_report': classification_report(y_test, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
    }


def cross_validate(model, X, y, cv=5, scoring='accuracy'):
    """Run cross-validation and return scores."""
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return {
        'scores': scores,
        'mean': np.mean(scores),
        'std': np.std(scores),
    }


def compare_models(models_dict, X_train, y_train, X_test, y_test):
    """Train and evaluate multiple models. Returns a results dict."""
    results = {}
    for name, model in models_dict.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        cv = cross_validate(model, X_train, y_train)
        results[name] = {
            'model': model,
            'test_metrics': metrics,
            'cv_mean': cv['mean'],
            'cv_std': cv['std'],
        }
    return results


def get_best_model(results):
    """Return (name, model) of the best model by CV accuracy."""
    best_name = max(results, key=lambda k: results[k]['cv_mean'])
    return best_name, results[best_name]['model']
