FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir .

# Config is yours to edit, so it is a bind mount; data and models are the
# container's to own. The example config reads these, so the same file works
# unchanged whether you run it here or from a checkout.
ENV HPMPC_CONFIG=/config/config.yaml \
    HPMPC_DATA_DIR=/data \
    HPMPC_MODEL_DIR=/data/models \
    HPMPC_STATE_FILE=/data/controller_state.json

VOLUME ["/data"]
EXPOSE 8129

HEALTHCHECK --interval=60s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8129/health', timeout=5).status == 200 else 1)"

ENTRYPOINT ["hpmpc"]
CMD ["serve"]
