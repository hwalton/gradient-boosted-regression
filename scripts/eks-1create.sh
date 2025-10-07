#!/usr/bin/env bash
set -euo pipefail

CLUSTER_NAME=${CLUSTER_NAME:-gbr-cluster4}
REGION=${REGION:-eu-west-2}
NODE_TYPE=${NODE_TYPE:-t3.small}   # change to t3.small or t3.medium if you accept charges
NODE_COUNT=${NODE_COUNT:-2}

echo "Creating EKS cluster ${CLUSTER_NAME} in ${REGION} with ${NODE_COUNT} x ${NODE_TYPE}..."
eksctl create cluster \
  --name "${CLUSTER_NAME}" \
  --region "${REGION}" \
  --nodes "${NODE_COUNT}" \
  --node-type "${NODE_TYPE}"