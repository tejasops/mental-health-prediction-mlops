"""Preprocessing pipeline for the Mental Health survey data."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Gender normalization maps
MALE_STR = [
    "male", "m", "male-ish", "maile", "mal", "male (cis)", "make",
    "male ", "man", "msle", "mail", "malr", "cis man", "cis male",
]
TRANS_STR = [
    "trans-female", "something kinda male?", "queer/she/they", "non-binary",
    "nah", "all", "enby", "fluid", "genderqueer", "androgyne", "agender",
    "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter",
    "female (trans)", "queer", "ostensibly male, unsure what that really means",
]
FEMALE_STR = [
    "cis female", "f", "female", "woman", "femake", "female ",
    "cis-female/femme", "female (cis)", "femail",
]
GENDER_GARBAGE = ["A little about you", "p"]


def drop_unnecessary_columns(df):
    """Drop columns not useful for modelling."""
    cols_to_drop = ['comments', 'state', 'Timestamp']
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


def fill_missing_values(df):
    """Fill NaN values with sensible defaults."""
    int_features = ['Age']
    string_features = [
        'Gender', 'Country', 'self_employed', 'family_history', 'treatment',
        'work_interfere', 'no_employees', 'remote_work', 'tech_company',
        'anonymity', 'leave', 'mental_health_consequence',
        'phys_health_consequence', 'coworkers', 'supervisor',
        'mental_health_interview', 'phys_health_interview',
        'mental_vs_physical', 'obs_consequence', 'benefits', 'care_options',
        'wellness_program', 'seek_help',
    ]

    for col in df.columns:
        if col in int_features:
            df[col] = df[col].fillna(0)
        elif col in string_features:
            df[col] = df[col].fillna('NaN')
    return df


def clean_gender(df):
    """Normalize the Gender column to male/female/trans."""
    def categorize_gender(g):
        g = str(g).lower().strip()
        if g in MALE_STR:
            return 'male'
        elif g in FEMALE_STR:
            return 'female'
        elif g in TRANS_STR:
            return 'trans'
        return g

    df['Gender'] = df['Gender'].apply(categorize_gender)
    df = df[~df['Gender'].isin(GENDER_GARBAGE)]
    return df


def clean_age(df):
    """Fix outlier ages and create age_range bins."""
    median_age = df['Age'].median()
    df['Age'] = df['Age'].fillna(median_age)
    df.loc[df['Age'] < 18, 'Age'] = median_age
    df.loc[df['Age'] > 120, 'Age'] = median_age

    df['age_range'] = pd.cut(
        df['Age'], [0, 20, 30, 65, 100],
        labels=["0-20", "21-30", "31-65", "66-100"],
        include_lowest=True,
    )
    return df


def clean_self_employed(df):
    """Replace NaN self_employed with No."""
    df['self_employed'] = df['self_employed'].replace(['NaN'], 'No')
    return df


def clean_work_interfere(df):
    """Replace NaN work_interfere with Don't know."""
    df['work_interfere'] = df['work_interfere'].replace(['NaN'], "Don't know")
    return df


def encode_features(df):
    """Label-encode all columns. Returns (encoded_df, label_dict)."""
    label_dict = {}
    encoders = {}
    for col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_dict[col] = list(le.classes_)
        encoders[col] = le
    return df, label_dict, encoders


def run_preprocessing(df):
    """Run the full preprocessing pipeline.

    Returns (processed_df, label_dict, encoders).
    """
    df = drop_unnecessary_columns(df)
    df = fill_missing_values(df)
    df = clean_gender(df)
    df = clean_age(df)
    df = clean_self_employed(df)
    df = clean_work_interfere(df)
    df = df.drop(columns=['Country'])
    df, label_dict, encoders = encode_features(df)
    return df, label_dict, encoders
