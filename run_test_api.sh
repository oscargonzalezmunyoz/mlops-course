#!/usr/bin/env bash
set -euo pipefail # stop the script when any step fails

curl -w "\n" http://localhost:8000/health
curl -w "\n"  http://localhost:8000/model-info
curl -w "\n"  -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"age":0.038,"sex":0.05,"bmi":0.0617,"bp":0.0219,"s1":-0.0442,"s2":-0.0348,"s3":-0.0434,"s4":-0.0026,"s5":0.0199,"s6":-0.0176}'