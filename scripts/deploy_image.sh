#!/usr/bin/env bash
set -euo pipefail

# Load .env
set -a
source .env
set +a

AWS_REGION=${AWS_REGION:-eu-west-2}  # default to eu-west-2 if AWS_REGION not set
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO=${ECR_REPO:-gbr-trainer}
IMAGE_TAG=${IMAGE_TAG:-latest}
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}"

aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$AWS_REGION" >/dev/null 2>&1 || \
  aws ecr create-repository --repository-name "$ECR_REPO" --region "$AWS_REGION" >/dev/null


aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
docker build -t "${ECR_REPO}:${IMAGE_TAG}" -f training/Dockerfile .
docker tag "${ECR_REPO}:${IMAGE_TAG}" "${ECR_URI}"
docker push "${ECR_URI}"