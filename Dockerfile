# Serving image for the Logistics Exception Engine.
#
# CPU-only by design. Training happens on the host GPU (see REPRODUCIBILITY.md);
# this container only runs inference, and a CUDA base would add several GB for
# no benefit to a reviewer running `docker compose up` on a laptop.
#
# Python 3.11 matches the training venv, so the checkpoint loads under the same
# torch/ultralytics versions it was written by.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WEIGHTS_PATH=/app/weights/best.pt \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

WORKDIR /app

# libgl1 and libglib2.0-0 are required by OpenCV even in the headless build;
# curl backs the healthcheck below.
RUN apt-get update && apt-get install --no-install-recommends -y \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first so code edits do not invalidate the pip layer.
# The CPU torch index keeps the image roughly 2 GB instead of roughly 7 GB.
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu \
       -r requirements.txt

COPY app/ ./app/
COPY scripts/ ./scripts/
COPY dataset/data.yaml ./dataset/data.yaml

# Non-root. Ultralytics writes a settings file at import, so give the runtime
# user a writable home and config dir.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/weights /app/sample_images /tmp/Ultralytics \
    && chown -R appuser:appuser /app /tmp/Ultralytics
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
