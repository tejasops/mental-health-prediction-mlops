"""Feature engineering utilities."""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def scale_age(df):
    """Scale the Age column using MinMaxScaler."""
    scaler = MinMaxScaler()
    df['Age'] = scaler.fit_transform(df[['Age']])
    return df, scaler


def get_feature_target_split(df, target='treatment'):
    """Split dataframe into features (X) and target (y)."""
    y = df[target]
    X = df.drop(columns=[target])
    return X, y


def split_data(X, y, test_size=0.3, random_state=0):
    """Train/test split."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def get_top_correlated_features(df, target='treatment', k=10):
    """Return the top-k features most correlated with the target."""
    corrmat = df.corr(numeric_only=True)
    cols = corrmat.nlargest(k, target)[target].index.tolist()
    return cols
