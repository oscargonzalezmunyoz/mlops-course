#!/usr/bin/env bash
set -euo pipefail # stop the script when any step fails

URL="${API_URL:-http://localhost:8000}"
INPUT_VALID='{"age":0.038,"sex":0.05,"bmi":0.0617,"bp":0.0219,"s1":-0.0442,"s2":-0.0348,"s3":-0.0434,"s4":-0.0026,"s5":0.0199,"s6":-0.0176}'
INPUT_OUT_OF_RANGE='{"age":50,"sex":0.05,"bmi":0.0617,"bp":0.0219,"s1":-0.0442,"s2":-0.0348,"s3":-0.0434,"s4":-0.0026,"s5":0.0199,"s6":-0.0176}'
INPUT_MISSING_FIELDS='{"age":0.038}'

# POST to /predict and check the HTTP status code
# Usage: expect_status <expected_code> <json_body> <description>

expect_status() {
  local code
  code=$(curl -sS -o /dev/null -w "%{http_code}" -X POST "$URL/predict" -H "Content-Type: application/json" -d "%2")
  if [[ "$code" != "$1" ]]; then
    echo "FAIL: $3 -> expected $1, got $code"
    exit 1
  fi
  echo "OK: $3 -> $code"
}

echo "GET /health"
curl -sS --fail -w "\n" "$URL/health"

echo "GET /model-info"
curl -sS --fail -w "\n" "$URL/model-info"

echo "POST /predict with valid input"
curl -sS --fail -w "\n" -X POST "$URL/predict" -H "Content-Type: application/json" -d "$INPUT_VALID"

echo "POST /predict with invalid inputs"
expect_status 422 "$INPUT_OUT_OF_RANGE" "age out of range"
expect_status 422 "$INPUT_MISSING_FIELDS" "missing fields"

echo "All API tests passed"