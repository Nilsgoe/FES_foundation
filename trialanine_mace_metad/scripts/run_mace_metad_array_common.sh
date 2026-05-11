#!/bin/bash
set -euo pipefail

project_dir=/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/trialanine_mace_metad
mkdir -p "${project_dir}/logs" "${project_dir}/outputs" "${project_dir}/structures"

: "${METAD_SYSTEM:?METAD_SYSTEM must be gas or solution}"

if [ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]; then
  model_a=off
  model_b=omol
else
  model_a=mh1
  model_b=polar
fi

env_for_model() {
  local model_key="$1"
  case "${model_key}" in
  off|omol)
    echo /fhi/home/ngoen/software/mace_cueq_venv
    ;;
  mh1)
    echo /fhi/home/ngoen/software/mace_cueq_venv
    ;;
  polar)
    echo /fhi/home/ngoen/software/biASE-venv
    ;;
  *)
    echo "ERROR: unsupported model key ${model_key}" >&2
    exit 1
    ;;
  esac
}

input_for_model() {
  local model_key="$1"
  if [ "${METAD_SYSTEM}" = "gas" ]; then
    echo "${project_dir}/structures/trialanine_gas_amber_start.extxyz"
  elif [ "${METAD_SYSTEM}" = "solution" ]; then
    echo "${project_dir}/structures/trialanine_solution_${model_key}_npt.extxyz"
  else
    echo "ERROR: unsupported METAD_SYSTEM=${METAD_SYSTEM}" >&2
    exit 1
  fi
}

mode_for_system() {
  if [ "${METAD_SYSTEM}" = "gas" ]; then
    echo gas_metad
  elif [ "${METAD_SYSTEM}" = "solution" ]; then
    echo solution_metad
  else
    echo "ERROR: unsupported METAD_SYSTEM=${METAD_SYSTEM}" >&2
    exit 1
  fi
}

run_model() {
  local model_key="$1"
  local gpu_id="$2"
  local mace_env input_path mode run_label driver
  mace_env=$(env_for_model "${model_key}")
  input_path=$(input_for_model "${model_key}")
  mode=$(mode_for_system)
  run_label="${METAD_SYSTEM}_${model_key}_metad_job${SLURM_ARRAY_JOB_ID}_task${SLURM_ARRAY_TASK_ID}_gpu${gpu_id}"
  driver="${project_dir}/scripts/run_mace_trialanine.py"

  if [ ! -f "${input_path}" ]; then
    echo "ERROR: required input not found for ${model_key}: ${input_path}" >&2
    exit 1
  fi

  (
    # shellcheck source=/dev/null
    source "${project_dir}/scripts/_activate_env.sh"
    activate_env "${mace_env}"

    export PYTHONPATH="/nexus/posix0/FHI-Theory/ngoen/biASE${PYTHONPATH:+:${PYTHONPATH}}"
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
    export CUDA_VISIBLE_DEVICES="${gpu_id}"

    python "${driver}" \
      --mode "${mode}" \
      --model-key "${model_key}" \
      --input "${input_path}" \
      --output-prefix "${project_dir}/outputs/${run_label}" \
      --steps "${METAD_STEPS:-1000000}" \
      --temperature "${TEMPERATURE_K:-300}" \
      --timestep-fs "${TIMESTEP_FS:-0.5}"
  ) &
}

run_model "${model_a}" 0
pid_a=$!
run_model "${model_b}" 1
pid_b=$!

sleep 5
nvidia-smi > "${project_dir}/logs/gpu_util_${METAD_SYSTEM}_metad_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt" 2>&1 || true

wait "${pid_a}"
rc_a=$?
wait "${pid_b}"
rc_b=$?

[ "${rc_a}" -ne 0 ] && echo "ERROR: ${METAD_SYSTEM} ${model_a} on GPU0 exited with code ${rc_a}" >&2
[ "${rc_b}" -ne 0 ] && echo "ERROR: ${METAD_SYSTEM} ${model_b} on GPU1 exited with code ${rc_b}" >&2

[ $((rc_a + rc_b)) -ne 0 ] && exit 1
exit 0
