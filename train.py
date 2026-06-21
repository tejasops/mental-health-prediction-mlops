"""Run the full preprocessing + training pipeline and save artifacts."""
import os
import sys
import pickle

# Ensure src is importable
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_raw_data
from src.preprocessing import run_preprocessing
from src.feature_engineering import scale_age, get_feature_target_split, split_data
from src.models import get_models, save_model
from src.evaluation import compare_models, get_best_model

def main():
    print("Loading raw data...")
    df = load_raw_data()
    print(f"  Raw shape: {df.shape}")

    print("Running preprocessing...")
    df, label_dict, encoders = run_preprocessing(df)
    print(f"  Processed shape: {df.shape}")

    # Save processed data
    processed_path = os.path.join('data', 'processed', 'survey_cleaned.csv')
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"  Saved processed data to {processed_path}")

    print("Scaling & splitting...")
    df_scaled, scaler = scale_age(df.copy())
    X, y = get_feature_target_split(df_scaled, target='treatment')
    feature_names = list(X.columns)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    print("Training models...")
    models = get_models()
    results = compare_models(models, X_train, y_train, X_test, y_test)

    print(f"\n{'Model':<25} {'Test Acc':>10} {'CV Mean':>10} {'CV Std':>10}")
    print('-' * 55)
    for name, res in sorted(results.items(), key=lambda x: -x[1]['cv_mean']):
        acc = res['test_metrics']['accuracy']
        print(f"{name:<25} {acc:>10.4f} {res['cv_mean']:>10.4f} {res['cv_std']:>10.4f}")

    best_name, best_model = get_best_model(results)
    print(f"\nBest model: {best_name}")

    # Save best model
    model_path = os.path.join('models', 'best_model.pkl')
    save_model(best_model, model_path)
    print(f"Saved model to {model_path}")

    # Save encoders (needed by the web app)
    encoders_path = os.path.join('models', 'encoders.pkl')
    with open(encoders_path, 'wb') as f:
        pickle.dump(encoders, f)
    print(f"Saved encoders to {encoders_path}")

    # Save feature names (needed by the web app)
    features_path = os.path.join('models', 'feature_names.pkl')
    with open(features_path, 'wb') as f:
        pickle.dump(feature_names, f)
    print(f"Saved feature names to {features_path}")

    print("\nDone! All artifacts saved.")

if __name__ == '__main__':
    main()
