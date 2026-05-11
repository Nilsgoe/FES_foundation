#!/bin/bash
set -euo pipefail

source ./config.env

chunks="${1:-10}"
steps_per_chunk="${2:-1000000}"
dependency=""

echo "Submitting ${RUN_NAME}: ${chunks} chunks x ${steps_per_chunk} steps"

for chunk in $(seq 0 $((chunks - 1))); do
    jobid=$(sbatch --parsable \
        ${dependency:+--dependency=afterany:${dependency}} \
        --job-name="${JOB_NAME}" \
        --export=ALL,CHUNK_ID="${chunk}",STEPS_PER_CHUNK="${steps_per_chunk}" \
        run_metad_viper.sh)
    echo "chunk=${chunk} jobid=${jobid} dependency=${dependency:-none}"
    dependency="${jobid}"
done
