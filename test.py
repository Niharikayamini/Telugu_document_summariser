import pytest
import json
from testapp import app  # your Flask app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# Test Homepage Load
def test_homepage(client):
    response = client.get('/')
    assert response.status_code == 200

# Test Summarization API with valid text
def test_summarize_valid(client):
    response = client.post('/summarize', json={"text": "ఇది ఒక చిన్న కథ"})
    assert response.status_code == 200
    assert "summary" in response.get_json()

# Test Summarization API with missing text
def test_summarize_empty(client):
    response = client.post('/summarize', json={})
    assert response.status_code == 400
    assert "error" in response.get_json()

# Test GET request on prediction route
def test_predict_get(client):
    response = client.get('/predict')
    assert response.status_code == 200
