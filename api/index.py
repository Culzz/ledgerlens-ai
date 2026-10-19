"""The one function Vercel serves.

The application itself is not written twice: this imports the same `app` that
`uvicorn app.main:app` serves locally, so the deployed routes and the local ones
are the same object. The only thing this file does is put `backend` on the import
path, because Vercel adds the project root and the package lives one directory
down.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.main import app

__all__ = ["app"]
