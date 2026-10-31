#!/usr/bin/env bash
# Run the full MLops pipeline locally, in the same order as the CI workflow
# Usage: ./run_pipeline.sh         -> run the pipeline
#        ./run_pipeline.sh --clean -> delete generated data and mlruns first

set -euo pipefail # stop the script when any step fails

if [[ "${1:-}" == "--clean" ]]
then
    echo ">>> Cleaning generated files"
    rm -rf mlruns data/raw/*.csv data/processed/*.csv
fi

steps=(
    src.data.load_data
    src.data.validate_data
    src.data.preprocess
    src.data.features
    src.models.train
    src.models.register_model
    src.models.evaluate_model    
)

for step in "${steps[@]}"
do
    echo ">>> python -m $step"
    python -m "$step"
done

echo ">>> Pipeline finished OK"