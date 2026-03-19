# ---------------------------------------------------------------------------
# Verra One — production container image
# ---------------------------------------------------------------------------
# Build:  docker build -t verra-one .
# Run:    docker run -e VERRA_HOME=/data -e OLLAMA_HOST=http://ollama:11434 \
#                    -p 8484:8484 verra-one
# ---------------------------------------------------------------------------

FROM python:3.13-slim

LABEL org.opencontainers.image.title="Verra One" \
      org.opencontainers.image.description="Your business data, one conversation away." \
      org.opencontainers.image.source="https://github.com/your-org/verra"

# System dependencies required by chromadb / pymupdf native extensions.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install verra-one from PyPI (pin the version in production).
# In local builds, copy the wheel instead:
#   COPY dist/verra_one-*.whl .
#   RUN pip install --no-cache-dir verra_one-*.whl
RUN pip install --no-cache-dir verra-one

# Atlas persists data to VERRA_HOME; mount a volume there in production.
ENV VERRA_HOME=/data \
    PYTHONUNBUFFERED=1

VOLUME ["/data"]

EXPOSE 8484

# Default: start the HTTP server on 0.0.0.0:8484.
CMD ["verra", "serve", "--host", "0.0.0.0", "--port", "8484"]
