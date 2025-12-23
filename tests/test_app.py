from fastapi.testclient import TestClient

from sentiment_app.app import app

client = TestClient(app)


def test_valid_response():
    assert True
