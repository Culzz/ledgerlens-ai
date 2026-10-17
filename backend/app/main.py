"""The HTTP app.

Serves the API and, when the frontend has been built, its files from the same
origin, so a deployment is one process behind one URL. During development the two
run separately and CORS is configured for that.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import ask, documents
from app.core.config import get_settings

DESCRIPTION = """\
Ask a question about a financial report.

The model is not asked for the answer. It is asked for a program, in the same DSL
the calculation engine replays, and the number that comes back is the engine's.
Every response carries the program, the steps it took and the rows it read, so an
answer can be checked rather than trusted. When the program will not run there is
no answer at all, and the reason is returned instead of a plausible figure.
"""


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="LedgerLens", version="0.1.0", description=DESCRIPTION)

    origins = [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.include_router(documents.router, prefix="/api")
    app.include_router(ask.router, prefix="/api")

    dist = Path(settings.frontend_dist)
    if dist.is_dir():
        _serve_frontend(app, dist)

    return app


def _serve_frontend(app: FastAPI, dist: Path) -> None:
    """Serve the built frontend, letting the client-side router own unknown paths."""
    assets = dist / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    index = dist / "index.html"

    @app.get("/", include_in_schema=False)
    def root() -> FileResponse:
        return FileResponse(index)

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        candidate = dist / path
        return FileResponse(candidate if candidate.is_file() else index)


app = create_app()
