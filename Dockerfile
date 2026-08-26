FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir .

# Data and models live on a volume so a rebuild never loses a trained model.
VOLUME ["/data"]
ENV HPMPC_CONFIG=/config/config.yaml

EXPOSE 8129

HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8129/health', timeout=5).status == 200 else 1)"

ENTRYPOINT ["hpmpc"]
CMD ["--config", "/config/config.yaml", "serve"]
