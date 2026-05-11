#!/bin/bash
set -euo pipefail

mkdir -p logs outputs

for model_key in off omol mh1 polar; do
  sbatch --job-name="metad_${model_key}" --export=ALL,MODEL_KEY="${model_key}" run_metad_raccoon.sh
done

echo "Submitted 4 azobenzene MetaD jobs. Update tracking.md with the job IDs."
