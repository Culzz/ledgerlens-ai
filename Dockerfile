# Two stages. The frontend is built with Node and then thrown away, so the image
# that ships carries the Python app and the compiled assets and no toolchain.
#
# The FinQA split is downloaded here rather than committed, because the repository
# does not redistribute the dataset. It is about four megabytes.

FROM node:22-alpine AS web

WORKDIR /web

# Lockfile first, so editing a component does not reinstall the tree.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-fund --no-audit

COPY frontend/tsconfig.json frontend/vite.config.ts frontend/index.html ./
COPY frontend/src ./src
RUN npm run build


FROM python:3.11-slim AS app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# The dependencies come from pyproject, and the optional extras do not: nothing in
# the serving path needs PyMuPDF or PyTorch. That arrives with the ingestion track.
COPY pyproject.toml ./
COPY backend ./backend
RUN pip install .

COPY scripts ./scripts
RUN python scripts/download_data.py

COPY --from=web /web/dist ./frontend/dist

# A fresh container has no .env, so the provider is configured through the
# environment. LLM_PROVIDER=fake is the default and needs no key.
ENV FINQA_DATA_DIR=./data/finqa \
    FRONTEND_DIST=./frontend/dist

# The port comes from the environment because the platforms that build this image
# disagree about which one to use: Render sets PORT, Hugging Face Spaces sets it to
# 7860. Nothing local needs to know.
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:' + __import__('os').environ.get('PORT', '8000') + '/api/health', timeout=2)"

# Shell form, so ${PORT} is expanded.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
