# syntax=docker/dockerfile:1

# -------- Base build stage --------
FROM python:3.12-slim AS base
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps for Postgres optional backend
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libpq-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# -------- Runtime stage --------
FROM python:3.12-slim AS runtime
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create non-root user
RUN useradd -ms /bin/bash appuser

# Copy installed site-packages from build stage
COPY --from=base /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=base /usr/local/bin /usr/local/bin

# Copy app code
COPY . /app

# Expose remote port (if enabled)
EXPOSE 8787

# Healthcheck calls the health_check tool via a tiny Python runner
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
 CMD python -c "import server; print(server.handle_health_check({}, server.get_server_singleton()))" || exit 1

# Default envs (override in compose)
ENV REMOTE_ENABLED=false \
    REMOTE_BIND=0.0.0.0:8787 \
    RATE_LIMIT_RPS=10 \
    RATE_LIMIT_BURST=20

# Graceful stop via SIGTERM is handled by uvicorn
CMD ["python", "-c", "import asyncio, remote_server; asyncio.run(remote_server.serve())"]

