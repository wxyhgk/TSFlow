"""Compatibility entry point for the composed HTTP API router."""

from app.api.routes import create_router


api_router = create_router()

__all__ = ["api_router"]

