#!/usr/bin/env bash
set -euo pipefail

# load env
set -a
source .env
set +a

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}"

# pull latest image then run python inside the image by overriding the entrypoint
docker pull "${ECR_URI}"
docker run --rm -it --entrypoint python "${ECR_URI}" -c "import sys, pandas; print('python', sys.executable); print('pandas', pandas.__version__)"