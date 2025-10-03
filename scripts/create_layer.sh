#!/usr/bin/env bash
set -euo pipefail

rm -f layer.zip
IMAGE="public.ecr.aws/lambda/python:3.10"

# build layer inside Lambda base image (correct ABI for compiled wheels)
docker run --rm --entrypoint /bin/sh -v "$(pwd):/out" "${IMAGE}" -c "\
  python -m pip install --no-cache-dir -r /out/requirements.layer.txt -t /tmp/python && \
  # prune caches/tests/pyc to reduce unzipped size
  find /tmp/python -name '__pycache__' -type d -exec rm -rf {} + || true; \
  find /tmp/python -name 'tests' -type d -exec rm -rf {} + || true; \
  find /tmp/python -name '*.pyc' -type f -delete || true; \
  python -m zipfile -c /out/layer.zip /tmp/python"
echo "layer.zip created at $(pwd)/layer.zip"