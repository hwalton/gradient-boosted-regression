#!/usr/bin/env bash
set -euo pipefail

FILE="scripts/${1:-call.json}"
URL="${SERVE_URL:-ac672901db4c7418f93dcc2a6deba02a-1263538825.eu-west-2.elb.amazonaws.com/predict}"

if [[ ! -f "$FILE" ]]; then
  echo "Error: payload file not found: $FILE" >&2
  exit 2
fi

echo "POST $URL with payload $FILE"
# prefer jq for pretty output if available
if command -v jq >/dev/null 2>&1; then
  curl -sS -X POST "$URL" -H "Content-Type: application/json" --data @"$FILE" | jq .
else
  curl -sS -X POST "$URL" -H "Content-Type: application/json" --data @"$FILE"
fi