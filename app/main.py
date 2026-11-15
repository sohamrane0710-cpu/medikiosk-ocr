from fastapi import FastAPI

app = FastAPI(
    title="Medikiosk OCR Service",
    description="Independent OCR microservice for prescription processing in the Medikiosk healthcare platform",
    version="0.1.0",
)


@app.get("/", summary="Root endpoint")
async def root() -> dict[str, str]:
    """Root endpoint providing service identification and status."""
    return {
        "status": "ok",
        "service": "Medikiosk OCR",
        "version": "0.1.0",
    }


@app.get("/health", summary="Basic health check")
async def health_check() -> dict[str, str]:
    """Health check endpoint used by deployment platforms (e.g., Render) and monitoring."""
    return {
        "status": "healthy",
        "service": "Medikiosk OCR",
    }


@app.get("/api/v1/health", summary="API v1 health check")
async def api_v1_health_check() -> dict[str, str]:
    """API v1 health check endpoint conforming to versioned API routing."""
    return {
        "status": "healthy",
        "service": "Medikiosk OCR",
        "api_version": "v1",
    }
