# Serving image. The trained model (artifacts/) is baked in so every image tag
# maps to exactly one model version.
FROM python:3.12-slim AS build
WORKDIR /build
COPY requirements.txt .
RUN python -m venv /venv && /venv/bin/pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim
ENV PATH="/venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/src MODEL_DIR=/app/artifacts
RUN useradd --uid 10001 --no-create-home appuser
WORKDIR /app
COPY --from=build /venv /venv
COPY src/ ./src/
COPY serving/ ./serving/
COPY artifacts/ ./artifacts/
USER 10001
EXPOSE 8000
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
