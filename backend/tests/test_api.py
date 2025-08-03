"""Basic API tests.

These tests exercise the FastAPI application using the TestClient to
ensure that the routes respond with the expected status codes and data
shapes.  They do not test the internal logic of agents or models.
"""

from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_health() -> None:
    """Verify that the health endpoint responds with a status message."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_summarisation_endpoint() -> None:
    """Verify that the summarisation endpoint returns a summary field."""
    payload = {"text": "The quick brown fox jumps over the lazy dog"}
    response = client.post("/api/v1/summarise", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "references" in data
    assert isinstance(data["references"], list)


def test_search_endpoint() -> None:
    """Verify that the search endpoint returns a list of results."""
    response = client.get("/api/v1/search", params={"q": "machine learning"})
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert isinstance(data["results"], list)
