#!/bin/bash
set -euo pipefail

mkdir -p logs outputs

for model_key in off omol mh1 polar; do
  sbatch --job-name="metad_${model_key}_cont" --export=ALL,MODEL_KEY="${model_key}" run_metad_continue_raccoon.sh
done
