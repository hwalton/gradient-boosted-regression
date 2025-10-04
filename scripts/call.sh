#!/usr/bin/env bash
set -euo pipefail

FILE="scripts/${1:-call.json}"
URL="${SERVE_URL:-http://127.0.0.1:8080/predict}"

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