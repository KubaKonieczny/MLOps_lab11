from fastapi.testclient import TestClient

from sentiment_app import app

client = TestClient(app)


def test_empty_string():
    response = client.post("/predict", json={"text": ""})

    assert response.status_code == 422
    assert response.json() == {"detail": "Input text should be a non-empty string"}


def test_valid_response():
    response = client.post("/predict", json={"text": "Test"})
    assert response.status_code == 200
    assert isinstance(response.json(), dict)
