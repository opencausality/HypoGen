"""FastAPI application factory for HypoGen."""

from __future__ import annotations

from fastapi import FastAPI

from hypogen.api.routes import router
from hypogen.config import HypoGenSettings


def create_app(settings: HypoGenSettings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Parameters
    ----------
    settings:
        Optional settings override. Uses ``get_settings()`` if not provided.

    Returns
    -------
    FastAPI
        Configured application ready to serve.
    """
    app = FastAPI(
        title="HypoGen API",
        description="Generate novel scientific hypotheses by mining causal gaps in research literature.",
        version="0.1.0",
    )

    app.include_router(router)

    return app
