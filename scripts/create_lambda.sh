#!/usr/bin/env bash

# get env vars
set -a
source .env
set +a

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}"

# create lambda function
aws lambda create-function \
  --function-name gbr-trainer \
  --role ${ROLE_ARN} \
  --package-type Image \
  --code ImageUri=${ECR_URI} \
  --timeout 900 --memory-size 2048 \
  --environment Variables="{MODEL_BUCKET=${BUCKET},PROCESSED_PREFIX=processed/,MODEL_KEY=models/gbr.joblib,GIT_REPO=${GIT_REPO},GIT_REF=master,FORCE_PULL=true}" \
  --region eu-west-2