#!/bin/bash
set -euo pipefail

mkdir -p logs outputs
sbatch run_opt_viper.sh
