"""Run the FastAPI interactive reranker server."""

from __future__ import annotations

import uvicorn

if __name__ == "__main__":
    print("Starting Multimodal Product Reranker Studio on http://127.0.0.1:8000 ...")
    uvicorn.run("mmrerank.app:app", host="127.0.0.1", port=8000, reload=True)
