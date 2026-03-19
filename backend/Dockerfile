# Backend Dockerfile — uses uv for fast, reproducible installs
# uv installs deps ~10-100× faster than pip and respects uv.lock exactly.

FROM python:3.11-slim

# Install uv (single binary, no extra deps)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# System deps (only what uv/the packages actually need at build time)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifest first so Docker layer-caches the install
# when only source code changes (not pyproject.toml / uv.lock).
COPY pyproject.toml uv.lock* ./

# Install production deps into the system Python — no venv needed in Docker.
# --frozen  : use uv.lock exactly, never re-resolve
# --no-dev  : skip pytest / dev tools
# --no-cache: keep image lean
RUN uv sync --frozen --no-dev --no-cache

# Now copy the rest of the source
COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
