#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME=continuous-control:latest

echo "Building Docker image $IMAGE_NAME..."
docker build -t $IMAGE_NAME .

mkdir -p checkpoints/demos

echo "Running container to generate demo artifacts..."
docker run --rm -v $(pwd)/checkpoints/demos:/app/checkpoints/demos $IMAGE_NAME --checkpoints checkpoints --out checkpoints/demos --fps 6

echo "Artifacts should be available in checkpoints/demos/"
