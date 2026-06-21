"""Data loading utilities for the Mental Health Prediction project."""
import os
import pandas as pd


def load_raw_data(filepath=None):
    """Load the raw survey CSV data."""
    if filepath is None:
        filepath = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'raw', 'survey.csv'
        )
    return pd.read_csv(filepath)


def load_processed_data(filepath=None):
    """Load the cleaned/processed survey CSV data."""
    if filepath is None:
        filepath = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'processed', 'survey_cleaned.csv'
        )
    return pd.read_csv(filepath)
