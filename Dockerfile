# =============================================================
# Dockerfile  —  SmartCity Traffic Control System
#
# HOW TO BUILD:
#   docker build -t smartcity-traffic:latest .
#
# HOW TO RUN:
#   docker run -p 8000:8000 smartcity-traffic:latest
#
# HOW TO PUSH TO HUGGINGFACE SPACES:
#   openenv push
# =============================================================

FROM ghcr.io/meta-pytorch/openenv-base:latest

# Set working directory inside the container
WORKDIR /app

# Copy requirements first (Docker caches this layer)
# If requirements don't change, this layer is not rebuilt
COPY server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Copy all project files into the container
COPY models.py              /app/models.py
COPY server/                /app/server/
COPY agent.py               /app/agent.py
COPY inference.py           /app/inference.py
COPY openenv.yaml           /app/openenv.yaml

# Health check — HuggingFace Spaces uses this
# Every 30 seconds, check if /health returns 200 OK
HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=10s \
    --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port 8000 — required for HuggingFace Spaces
EXPOSE 8000

# Start the FastAPI server when container launches
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
