"""Aggregate router preserving the public `/jobs` API."""

from fastapi import APIRouter

from app.api.routes.job_routes import creation, downloads, events, queries


router = APIRouter()
router.include_router(creation.router, prefix="/jobs")
router.include_router(events.router, prefix="/jobs")
router.include_router(downloads.router, prefix="/jobs")
router.include_router(queries.router, prefix="/jobs")
