#!/usr/bin/env bash
set -euo pipefail

RELEASE=${RELEASE:-gbr}
CHART=${CHART:-helm/gbr}
NAMESPACE=${NAMESPACE:-default}
VALUES_FILE=${VALUES_FILE:-helm/gbr/values.yaml}
TIMEOUT=${TIMEOUT:-10m}
# Optional override for image tag
IMAGE_ML_TAG=${IMAGE_ML_TAG:-}
IMAGE_AIRFLOW_TAG=${IMAGE_AIRFLOW_TAG:-}

echo "Linting Helm chart ${CHART}..."
helm lint "${CHART}"

echo "Ensuring namespace ${NAMESPACE} exists..."
kubectl get namespace "${NAMESPACE}" >/dev/null 2>&1 || kubectl create namespace "${NAMESPACE}"

echo "Deploying Helm release ${RELEASE} -> chart ${CHART} in namespace ${NAMESPACE}..."
SET_ARGS=()
[ -n "${IMAGE_ML_TAG}" ] && SET_ARGS+=(--set image.ml.tag="${IMAGE_ML_TAG}")
[ -n "${IMAGE_AIRFLOW_TAG}" ] && SET_ARGS+=(--set image.airflow.tag="${IMAGE_AIRFLOW_TAG}")

helm upgrade --install "${RELEASE}" "${CHART}" \
  -n "${NAMESPACE}" \
  -f "${VALUES_FILE}" \
  "${SET_ARGS[@]}" \
  --wait --timeout "${TIMEOUT}"

echo "Deployment finished. Showing release status:"
helm status "${RELEASE}" -n "${NAMESPACE}"