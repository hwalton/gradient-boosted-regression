#!/usr/bin/env bash
set -euo pipefail

# Run from repo root; switch into infrastructure so CDK finds cdk.json
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# change into infrastructure where cdk.json and app.py live
cd infrastructure

# activate infra venv if present
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo ".venv not found in infrastructure/, creating one..."
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install --upgrade pip setuptools wheel
  pip install -r requirements.txt
fi

: "${AWS_REGION:=eu-west-2}"
export AWS_REGION

JSII_SILENCE_WARNING_UNTESTED_NODE_VERSION=1
export JSII_SILENCE_WARNING_UNTESTED_NODE_VERSION

echo "Bootstrapping CDK in account/region (may prompt)..."
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
cdk bootstrap aws://${ACCOUNT}/${AWS_REGION}

echo "Synthesizing..."
cdk synth

echo "Deploying (stack: InfrastructureStack10)..."
cdk deploy InfrastructureStack10 --require-approval never