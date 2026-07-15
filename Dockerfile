FROM python:3.10-slim

WORKDIR /app

# System deps (optional but nice-to-have for faster wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source
COPY . .

# Environment (OpenAI key picked up from env or .env if you mount it)
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

# Default entry: the unified dashboard bundling Phases 1-4 (can be overridden,
# e.g. by scripts/run_cli.sh for the QMSum batch runner)
CMD ["python", "app/dashboard.py"]
