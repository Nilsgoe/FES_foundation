#!/bin/bash
set -euo pipefail

mkdir -p logs
echo "Submitting malonaldehyd MACE-OFF24 umbrella MD"
MODEL_FAMILY=off MODEL_SIZE=off24 \
  sbatch --job-name=umd_off24 run_U_MDs_cueq.sh
