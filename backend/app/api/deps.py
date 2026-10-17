"""Shared objects the routes need, built once per process."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import HTTPException, status

from app.core.config import get_settings
from app.services.library import DocumentLibrary
from app.services.llm import LLMProvider, build_provider
from app.services.reasoning import Orchestrator


@lru_cache
def get_library() -> DocumentLibrary:
    settings = get_settings()
    return DocumentLibrary(Path(settings.finqa_data_dir) / "dev.json")


def require_library() -> DocumentLibrary:
    """The library, or a clear 503 when the split has not been downloaded."""
    library = get_library()
    if not library.available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No documents loaded. Run `python scripts/download_data.py` to fetch "
                "the FinQA split into the data directory."
            ),
        )
    return library


@lru_cache
def get_provider() -> LLMProvider:
    return build_provider(get_settings())


@lru_cache
def get_orchestrator() -> Orchestrator:
    settings = get_settings()
    return Orchestrator(get_provider(), max_tokens=settings.llm_max_output_tokens or 1024)
