"""Unit tests for the FastAPI reranker application."""

from __future__ import annotations

from fastapi.testclient import TestClient

from mmrerank.app import app


def test_index_page() -> None:
    """Test that the index endpoint returns HTML."""
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "MM-Rerank Studio" in response.text


def test_sample_queries_api() -> None:
    """Test sample queries endpoint."""
    with TestClient(app) as client:
        response = client.get("/api/sample_queries")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        if len(data) > 0:
            assert "query" in data[0]
            assert "products" in data[0]


def test_rerank_api() -> None:
    """Test reranking endpoint."""
    payload = {
        "query": "silk bodysuit long sleevetops",
        "products": [
            {
                "product_id": "p1",
                "title": "Women Silk Long Sleeve Bodysuit",
                "esci_label": "E",
                "has_image": 1,
                "clip_similarity": 0.85,
            },
            {
                "product_id": "p2",
                "title": "Men Cotton T-Shirt",
                "esci_label": "I",
                "has_image": 1,
                "clip_similarity": 0.20,
            },
        ],
        "model_variant": "text_plus_image",
    }
    with TestClient(app) as client:
        response = client.post("/api/rerank", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == payload["query"]
        assert len(data["ranked_products"]) == 2
        assert "ndcg_10" in data
        assert data["ranked_products"][0]["product_id"] == "p1"


def test_compare_api() -> None:
    """Test comparison endpoint."""
    payload = {
        "query": "pool cover",
        "products": [
            {
                "product_id": "p1",
                "title": "18x34 Heavy Duty Winter Pool Cover",
                "esci_label": "E",
                "has_image": 1,
                "clip_similarity": 0.90,
            },
            {
                "product_id": "p2",
                "title": "Pool Chemicals Chlorine Tablets",
                "esci_label": "S",
                "has_image": 0,
                "clip_similarity": None,
            },
        ],
    }
    with TestClient(app) as client:
        response = client.post("/api/compare", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "text_only" in data
        assert "text_plus_image" in data
        assert "delta_ndcg" in data
        assert len(data["text_plus_image"]) == 2
