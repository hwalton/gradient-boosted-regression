#!/usr/bin/env bash
set -euo pipefail

# Validate payload
if ! jq . training/payload.json >/dev/null 2>&1; then
  echo "Invalid JSON payload: training/payload.json" >&2
  exit 2
fi

# Invoke with correct binary format for AWS CLI v2
aws lambda invoke \
  --function-name gbr-trainer \
  --payload fileb://training/payload.json \
  --cli-binary-format raw-in-base64-out \
  response.json || true

# Print response
jq . response.json || cat response.json

# show recent CloudWatch logs to help diagnose import errors
aws logs filter-log-events --log-group-name /aws/lambda/gbr-trainer --limit 50 --query 'events[].{time:timestamp,message:message}' --output json | jq .