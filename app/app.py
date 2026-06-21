"""Flask REST API for Mental Health Treatment Prediction."""
import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pkl')
ENCODERS_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'encoders.pkl')
FEATURE_NAMES_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'feature_names.pkl')

model = None
encoders = None
feature_names = None


def load_artifacts():
    global model, encoders, feature_names
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(ENCODERS_PATH, 'rb') as f:
        encoders = pickle.load(f)
    with open(FEATURE_NAMES_PATH, 'rb') as f:
        feature_names = pickle.load(f)


def encode_input(data):
    """Convert a JSON dict to model input DataFrame using stored encoders."""
    try:
        age = int(data.get('Age', 30))
    except (ValueError, TypeError):
        age = 30

    if age <= 20:
        age_range = '0-20'
    elif age <= 30:
        age_range = '21-30'
    elif age <= 65:
        age_range = '31-65'
    else:
        age_range = '66-100'

    row = {}
    for feat in feature_names:
        if feat == 'age_range':
            val = age_range
        elif feat == 'Age':
            val = str(age)
        else:
            val = data.get(feat, '')
            if not val:
                val = encoders[feat].classes_[0]

        le = encoders[feat]
        if val in le.classes_:
            row[feat] = le.transform([val])[0]
        else:
            row[feat] = 0

    return pd.DataFrame([row], columns=feature_names)


@app.route('/health', methods=['GET'])
def health():
    """Used by Kubernetes liveness and readiness probes."""
    if model is None:
        return jsonify({'status': 'unhealthy', 'reason': 'model not loaded'}), 503
    return jsonify({'status': 'healthy'}), 200


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'model not loaded'}), 503

    data = request.get_json(silent=True, force=True)
    if data is None:
        return jsonify({'error': 'request body must be JSON'}), 400

    try:
        input_df = encode_input(data)
        prediction = model.predict(input_df)[0]
        treatment_le = encoders['treatment']
        result = treatment_le.inverse_transform([prediction])[0]

        return jsonify({
            'prediction': result,
            'recommends_treatment': bool(result == 'Yes')
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'service': 'mental-health-prediction-api',
        'version': '1.0.0',
        'endpoints': ['/health', '/predict']
    }), 200


load_artifacts()

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)