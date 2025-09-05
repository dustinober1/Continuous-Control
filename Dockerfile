FROM python:3.10-slim

# Install system deps for matplotlib and image handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Default entrypoint is to generate plots (safe headless action)
ENTRYPOINT ["python3", "scripts/generate_plot.py"]
