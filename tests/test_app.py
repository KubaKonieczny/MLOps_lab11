from fastapi.testclient import TestClient

from sentiment_app.app import app

client = TestClient(app)


def test_valid_response():
    response = client.post("/predict", json={"text": "Test"})
    assert response.status_code == 200
    assert isinstance(response.json(), dict)
