set -a; source .env; set +a
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}"

# quick test (runs python inside the built image)
docker run --rm --entrypoint python "${ECR_URI}" -c "import importlib.util,sys; print('spec', importlib.util.find_spec('pandas')); import pandas; print('pandas', pandas.__version__)"