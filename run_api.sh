#!/usr/bin/env bash
docker build -t mlops-api .
docker run --rm -p 8000:8000 mlops-api