"""Tests for the mental health prediction Flask API."""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.app import app


@pytest.fixture
def client():
    """Flask's test client - sends fake requests without running a real server."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_health_check(client):
    """The /health endpoint should report healthy once the model is loaded."""
    response = client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'healthy'


def test_index_returns_service_info(client):
    """The root endpoint should describe the service, not render HTML."""
    response = client.get('/api')
    assert response.status_code == 200
    data = response.get_json()
    assert data['service'] == 'mental-health-prediction-api'
    assert '/predict' in data['endpoints']


def test_predict_with_valid_data(client):
    """A well-formed prediction request should return a valid result."""
    payload = {
        'Age': 28,
        'Gender': 'male',
        'family_history': 'Yes',
        'work_interfere': 'Often'
    }
    response = client.post('/predict', json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert 'prediction' in data
    assert data['prediction'] in ['Yes', 'No']
    assert isinstance(data['recommends_treatment'], bool)


def test_predict_with_missing_fields_still_works(client):
    """Missing fields should fall back to encoder defaults, not crash."""
    payload = {'Age': 35}
    response = client.post('/predict', json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert 'prediction' in data


def test_predict_with_no_body_returns_400(client):
    """Sending no JSON body at all should be rejected cleanly, not crash."""
    response = client.post('/predict')
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data


def test_predict_with_invalid_age_falls_back_to_default(client):
    """A garbage Age value should not crash the request."""
    payload = {'Age': 'not-a-number', 'Gender': 'female'}
    response = client.post('/predict', json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert 'prediction' in data