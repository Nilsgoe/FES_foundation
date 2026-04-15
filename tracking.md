# MACE Umbrella MD — Job Tracking

Umbrella MD: shifts −5 to 35 (step 0.05 Å), 41 windows, n_steps = 50 000, T = 293 K.  
Two GPUs per job (A100 80 GB), SLURM array 0–20 %1 (21 tasks × 2 shifts).

---

## f-malonaldehyd

**Initial geometry:** `f-malonaldehyd/optimized_fmalonaldehyde_initial.xyz`  
**CV:** d(O4–H8) − d(O3–H8)  
**Output path:** `f-malonaldehyd/outputs/`

| Model | Size / key | Status | Submitted | Job ID(s) | Venv | Notes |
|-------|-----------|--------|-----------|-----------|------|-------|
| mace-off | small | ✅ done | ~2026-04-02 | 27318 (array) | biASE-venv | cueq OFF |
| mace-off | medium | ✅ done | ~2026-04-02 | 27318 (array) | biASE-venv | cueq OFF |
| mace-off | large | ✅ done | ~2026-04-02 | 27318 (array) | biASE-venv | cueq OFF |
| mace-omol | extra_large | ✅ done | ~2026-04-08 | 27329–27330 | biASE-venv | cueq OFF |
| mace-polar | polar-1-s | ⏳ pending | — | — | mace_alpha_cueq_venv | cueq ON |
| mace-polar | polar-1-m | ⏳ pending | — | — | mace_alpha_cueq_venv | cueq ON |
| mace-polar | polar-1-l | ⏳ pending | — | — | mace_alpha_cueq_venv | cueq ON |
| mace-mh1 | mh-1 | ⏳ pending | — | — | mace_alpha_cueq_venv | cueq ON |

---

## malonaldehyd

**Initial geometry:** `malonaldehyd/optimized_malonaldehyde_initial.xyz`  
**CV:** d(O4–H8) − d(O3–H8)  
**Output path:** `malonaldehyd/outputs/`

| Model | Size / key | Status | Submitted | Job ID(s) | Venv | Notes |
|-------|-----------|--------|-----------|-----------|------|-------|
| mace-off | small | ✅ done | ~2026-04-01 | 27241 (array) | biASE-venv | cueq OFF |
| mace-off | medium | ✅ done | ~2026-04-01 | 27241 (array) | biASE-venv | cueq OFF |
| mace-off | large | ✅ done | ~2026-04-01 | 27241 (array) | biASE-venv | cueq OFF |
| mace-omol | extra_large | ✅ done | ~2026-04-07 | 26981–26984 | biASE-venv | cueq OFF |
| mace-polar | polar-1-s | ⏳ pending | — | — | mace_alpha_cueq_venv | cueq ON |
| mace-polar | polar-1-m | ⏳ pending | — | — | mace_alpha_cueq_venv | cueq ON |
| mace-polar | polar-1-l | ⏳ pending | — | — | mace_alpha_cueq_venv | cueq ON |
| mace-mh1 | mh-1 | ⏳ pending | — | — | mace_alpha_cueq_venv | cueq ON |

---

## How to submit new jobs

SSH to fhi-raccon, then:

```bash
# f-malonaldehyd (all 4 new models at once)
cd /nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/f-malonaldehyd
bash submit_new_models.sh

# malonaldehyd
cd /nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/malonaldehyd
bash submit_new_models.sh
```

Each `submit_new_models.sh` queues 4 SLURM array jobs (polar-s, polar-m, polar-l, mh1).  
Wall time per array job: **12 h** (doubled from old 6 h).  
Update the job IDs in this file after submission.

---

## Output file naming convention

```
outputs/umd_raccoon_{family}_{size}_shift_{N}_{run_label}_{gpu}.traj
outputs/cv_energy_raccoon_{family}_{size}_shift_{N}_{run_label}_{gpu}.csv
outputs/mean_cv_energy_raccoon_{family}_{size}_shift_{N}_{run_label}_{gpu}.csv
outputs/bfgs_raccoon_{family}_{size}_shift_{N}_{run_label}_{gpu}.log
```

Examples for new models:
- `umd_raccoon_polar_l_shift_0_raccoon_polar_l_job12345_task2_gpu0.traj`
- `mean_cv_energy_raccoon_mh1_mh-1_shift_5_raccoon_mh1_mh-1_job12346_task5_gpu1.csv`
