"""Health and basic endpoint verification tests."""

import json
import threading
import time
import urllib.request
import uvicorn
from app.main import app


def test_endpoints_live():
    """Verify all health and documentation endpoints serve HTTP 200."""
    port = 8765
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    time.sleep(1)

    try:
        # Test Root Endpoint
        req_root = urllib.request.urlopen(f"http://127.0.0.1:{port}/")
        assert req_root.status == 200
        root_data = json.loads(req_root.read().decode())
        assert root_data["status"] == "ok"
        assert root_data["service"] == "Medikiosk OCR"

        # Test Health Endpoint
        req_health = urllib.request.urlopen(f"http://127.0.0.1:{port}/health")
        assert req_health.status == 200
        health_data = json.loads(req_health.read().decode())
        assert health_data["status"] == "healthy"
        assert health_data["service"] == "Medikiosk OCR"

        # Test API v1 Health Endpoint
        req_v1_health = urllib.request.urlopen(f"http://127.0.0.1:{port}/api/v1/health")
        assert req_v1_health.status == 200
        v1_data = json.loads(req_v1_health.read().decode())
        assert v1_data["status"] == "healthy"
        assert v1_data["api_version"] == "v1"

        # Test Swagger Docs Endpoint
        req_docs = urllib.request.urlopen(f"http://127.0.0.1:{port}/docs")
        assert req_docs.status == 200
        assert "text/html" in req_docs.headers.get("content-type", "")

        # Test OpenAPI JSON Endpoint
        req_openapi = urllib.request.urlopen(f"http://127.0.0.1:{port}/openapi.json")
        assert req_openapi.status == 200
        openapi_data = json.loads(req_openapi.read().decode())
        assert openapi_data["info"]["title"] == "Medikiosk OCR Service"

    finally:
        server.should_exit = True
        thread.join(timeout=2)


if __name__ == "__main__":
    test_endpoints_live()
    print("All endpoint tests passed successfully!")
