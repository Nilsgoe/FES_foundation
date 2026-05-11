# Trialanine MACE Workflow

Project path: `/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/trialanine_mace_metad`

Target system: capped trialanine, `ACE-ALA-ALA-ALA-NME` (`AcAla3NMe`), in gas phase and explicit TIP3P water.

Current active state, 2026-05-11:
- The previous `ACE-ALA-ALA-NME` setup was identified as the wrong system (`AcAla2NMe`).
- The old generated simulation artifacts were archived to `archive/wrong_acala2nme_2026-05-08/`.
- The active project directories `structures/`, `outputs/`, and `logs/` were reset for a clean rebuild.
- `setup_amber/tleap.in` now builds the corrected `ACE-ALA-ALA-ALA-NME` peptide.
- Corrected Amber setup completed as job `44109`.
- Corrected MACE solution equilibration was submitted as array job `44110`.
- Current corrected solution-equilibration status:
  - `off`: NVT and NPT complete
  - `mh1`: NVT and NPT complete
  - `polar`: NVT and NPT complete
  - `omol`: NVT complete, NPT incomplete
- Corrected gas-phase MetaD was submitted on `fhi-raccoon` as job `44145`, but is still pending on queue priority.
- Corrected gas-phase MetaD was also submitted on `viper-gpu`:
  - `pet-spice`: 2-chunk `1 ns` chain, chunk 0 completed and chunk 1 running
  - `sol3r`: original 2-chunk `1 ns` chain failed; patched rerun submitted as `8606309 -> 8606310`

Important note:
- Historical entries below describe the archived `AcAla2NMe` attempt and should not be used as the active state for this project anymore.

## 1. AmberTools Starting Structures

Run file: `scripts/run_amber_setup.slurm`

Environment: `/fhi/home/ngoen/software/ambertools_setup_env`

Purpose:
- AmberTools is used only for peptide construction, solvation, minimization, heating, and short starting-structure equilibration.
- Amber is not used for production MD, MetaD, or final density determination.
- The Amber solution box is only an initial guess for MACE NPT.

Inputs:
- `setup_amber/tleap.in`
- `setup_amber/min.in`
- `setup_amber/heat.in`
- `setup_amber/equil_nvt.in`
- `setup_amber/equil_npt.in`
- Gas-specific files: `setup_amber/min_gas.in`, `setup_amber/heat_gas.in`, `setup_amber/equil_gas.in`

Corrected Amber setup details:
- Force field: `leaprc.protein.ff14SB`, used because AmberTools 24 on `fhi-raccoon` does not provide `leaprc.protein.ff99SB`.
- Solvent: TIP3P water.
- Solvation: `631` waters from `solvateBox ... 4.6 iso`.
- Gas minimization: `2000` minimization cycles.
- Gas heating: `10000` steps, `dt = 0.001 ps = 1 fs`, `0 -> 300 K`, NVT.
- Gas equilibration: `20000` steps, `dt = 0.001 ps = 1 fs`, `300 K`, NVT.
- Solution minimization: `2000` minimization cycles.
- Solution heating: `10000` steps, `dt = 0.001 ps = 1 fs`, `0 -> 300 K`, NVT.
- Solution NVT: `20000` steps, `dt = 0.001 ps = 1 fs`, `300 K`, total `20 ps`.
- Solution NPT: `100000` steps, `dt = 0.001 ps = 1 fs`, `293 K`, `1 bar`, total `100 ps`, `taup = 2 ps`.

Corrected Amber outputs used downstream:
- Gas: `structures/trialanine_gas_amber_start.extxyz`
- Solution: `structures/trialanine_solution_amber_start.extxyz`
- Reference PDBs: `structures/trialanine_gas_initial.pdb`, `structures/trialanine_solution_initial.pdb`
- Gas extxyz atom count: `42`
- Solution extxyz atom count: `1935`

## 2. MACE Solution NVT/NPT

Run file: `scripts/run_mace_solution_equil_array.slurm`

Corrected active job:
- Slurm job `44110`.
- `44110_0`: `GPU0 -> off`, `GPU1 -> omol`.
- `44110_1`: `GPU0 -> mh1`, `GPU1 -> polar`.

Environments:
- `off`: `/fhi/home/ngoen/software/mace_cueq_venv`
- `omol`: `/fhi/home/ngoen/software/mace_cueq_venv`
- `mh1`: `/fhi/home/ngoen/software/mace_cueq_venv`
- `polar`: `/fhi/home/ngoen/software/biASE-venv`

April 28, 2026 cueq validation for `mh1`:
- Verified interactively on `gpubig` that `mace_mp(model="mh-1", head="omol", enable_cueq=True, default_dtype="float32")` computes forces and energy for the full solvated trialanine system on an A100.
- Verified the patched shared driver `scripts/run_mace_trialanine.py` completes a short `solution_npt` canary (`10` NVT steps plus `10` NPT steps) with `mh1` in `/fhi/home/ngoen/software/mace_cueq_venv`.

May 1, 2026 cueq validation for `polar`:
- Saved a rollback snapshot of the pre-cueq `polar` environment at `/fhi/home/ngoen/software/venv_backups/biASE-venv_pre_polar_cueq_2026-05-01/`.
- Installed `cuequivariance==0.9.1`, `cuequivariance-torch==0.9.1`, and `cuequivariance-ops-torch-cu12==0.9.1` into `/fhi/home/ngoen/software/biASE-venv`.
- Verified interactively on `gpubig` that `mace_polar(model="polar-1-l", enable_cueq=True, device="cuda")` computes energy and forces for the full solvated trialanine system.
- Updated `scripts/run_mace_trialanine.py` so the `polar` calculator path now uses `enable_cueq=True`.

Per-model workflow:
- Input: `structures/trialanine_solution_amber_start.extxyz`
- Optional minimization: BFGS, `fmax = 0.05 eV/A`, maximum `500` steps.
- Fixed-cell MACE NVT.
- Write `structures/trialanine_solution_<model>_nvt.extxyz`.
- Variable-cell MACE NPT.
- Write `structures/trialanine_solution_<model>_npt.extxyz`.

MACE NVT/NPT physical constants:
- Device: `cuda`
- Timestep: `0.5 fs`
- Temperature: `293 K`
- Pressure: `1 bar`
- NVT steps: `40000`, total `20 ps`
- NPT steps: `200000`, total `100 ps`
- Total per-potential equilibration time: `120 ps`
- NVT thermostat: ASE `Langevin`, friction `0.1 fs^-1` as passed to ASE.
- NPT method: ASE `NPTBerendsen`
- NPT thermostat coupling `taut`: `1000 fs = 1 ps`
- NPT pressure coupling `taup`: `2000 fs = 2 ps`
- NPT compressibility: `4.57e-5 bar^-1`

Corrected solution-equilibration outputs now on disk:
- `structures/trialanine_solution_off_npt.extxyz`
- `structures/trialanine_solution_mh1_npt.extxyz`
- `structures/trialanine_solution_polar_npt.extxyz`
- `structures/trialanine_solution_omol_nvt.extxyz`

Current corrected solution status:
- `off`: NVT complete, NPT complete to `100 ps`
- `mh1`: NVT complete, NPT complete to `100 ps`
- `polar`: NVT complete, NPT complete to `100 ps`
- `omol`: NVT complete, NPT started but only reached about `0.50 ps`

NPT continuation path:
- `scripts/run_mace_solution_npt_continue_small.slurm`
- Uses `run_mace_trialanine.py --mode solution_npt_continue`
- Reads the last frame of a partial NPT trajectory using the ASE `@-1` syntax.
- If the final trajectory frame is truncated, the driver now walks backward to the last readable frame.
- Runs NPT only, without rerunning NVT.

## 3. Later MACE MetaD

Run files:
- Gas paired array: `scripts/run_mace_gas_metad_array.slurm`
- Solution paired array: `scripts/run_mace_solution_metad_array.slurm`
- Shared implementation: `scripts/run_mace_metad_array_common.sh`

Current corrected MetaD status:
- Solution MetaD is still intentionally not started, because `omol` does not yet have a corrected `*_npt.extxyz`.
- Gas MetaD was submitted on `fhi-raccoon` as job `44145`, but has not started yet because of queue priority.
- Gas MetaD was also set up on `viper-gpu` for `sol3r` and `pet-spice`.

Gas MetaD input:
- `structures/trialanine_gas_amber_start.extxyz`

Solution MetaD inputs:
- `structures/trialanine_solution_off_npt.extxyz`
- `structures/trialanine_solution_omol_npt.extxyz`
- `structures/trialanine_solution_mh1_npt.extxyz`
- `structures/trialanine_solution_polar_npt.extxyz`

MetaD physical constants in `scripts/run_mace_trialanine.py`:
- Timestep: `0.5 fs`
- Temperature: `293 K` by default after the 2026-04-23 update.
- Default MetaD steps: `1000000`, total `500 ps`
- Bias height: `0.1 eV`
- Gaussian widths: `[5 deg, 5 deg]`
- Bias deposition interval: `100` MD steps, every `50 fs`
- Well-tempered bias factor: `10`
- Bounds: `((-180 deg, 180 deg), (-180 deg, 180 deg))`
- Wrapping: `[True, True]`

Collective variables for corrected `AcAla3NMe`:
- `phi`: `C(ALA2)-N(ALA3)-CA(ALA3)-C(ALA3)`
- `psi`: `N(ALA3)-CA(ALA3)-C(ALA3)-N(ALA4)`
- Default 0-based ASE indices: `phi = (14, 16, 18, 24)`, `psi = (16, 18, 24, 26)`.

Validation:
- Script: `scripts/validate_cv_indices.py`
- Validation command: `python scripts/validate_cv_indices.py --pdb structures/trialanine_gas_initial.pdb --compare-solution-pdb structures/trialanine_solution_initial.pdb`
- Prior GPU-node validation job: `33995`.
- Result: gas and solution peptide atom ordering match for all `42` peptide atoms of corrected `AcAla3NMe`.

## 4. Gas MetaD Submission Status

### `fhi-raccoon` corrected MACE gas MetaD

Run file:
- `scripts/run_mace_gas_metad_array.slurm`

Submission:
- Job `44145`
- Array layout:
  - task `0`: `off` + `omol`
  - task `1`: `mh1` + `polar`

Current state, 2026-05-11:
- `44145_[0-1%2]` is still `PENDING (Priority)`
- No new corrected gas-phase raccoon MetaD outputs from `44145` exist yet

### `viper-gpu` corrected gas MetaD

Corrected gas-phase structures and validator were synced to:
- `/ptmp/ngoen/Documents/Enhanced_sampling/sol3r/trialanine/gas`
- `/ptmp/ngoen/Documents/Enhanced_sampling/upet/trialanine/gas_pet_spice`

Validation:
- Both Viper gas directories pass the corrected `AcAla3NMe` phi/psi validation with:
  - `phi = (14, 16, 18, 24)`
  - `psi = (16, 18, 24, 26)`

`pet-spice` gas MetaD:
- Run name: `pet_spice_trialanine_gas_phi_psi`
- Submission chain:
  - `8582707 -> 8582709`
- Total target length: `1 ns`
- Status:
  - `8582707`: completed
  - `8582709`: running

`sol3r` gas MetaD:
- Original chain:
  - `8582708 -> 8582710`
- Original outcome:
  - chunk 0 segfaulted
  - chunk 1 failed while ASE attempted to reopen/append the very large trajectory
- Diagnosis:
  - unlike the working SO3LR azobenzene workflow, the original trialanine SO3LR driver did not pass an explicit MetaD trajectory `loginterval`
  - this caused effectively every-step trajectory writing and a multi-GB `.traj`
- Fix:
  - patched Viper SO3LR gas driver now uses `trajectory-loginterval = 10`
  - failed outputs were archived on Viper before rerun
- Patched rerun chain:
  - `8606309 -> 8606310`
- Current state:
  - `8606309`: pending
  - `8606310`: pending on dependency

## 5. Interrupted Runs Archived

The following interrupted or obsolete attempts were moved out of active paths:
- `interrupted_runs/33986/`
- `interrupted_runs/33989/`
- `interrupted_runs/33991/`

Reason:
- These runs were cancelled or superseded before producing a complete set of valid model-specific NPT structures.
- Their files should not be used as production inputs.

Do not use these archived files for MetaD:
- Any `outputs/solution_*_eq_job33991_*`
- Any `structures/trialanine_solution_*_nvt.extxyz` inside `interrupted_runs/33991/`

## 6. Small-GPU Benchmarking

Benchmark file:
- `scripts/run_mace_solution_benchmark_array.slurm`

Purpose:
- Short solvated-trialanine benchmarking only.
- Runs one model per `gpusmall` task for all four models: `off`, `omol`, `mh1`, `polar`.
- Uses benchmark-specific output names so it does not overwrite the active production NVT/NPT structures.

Benchmark physical settings:
- System: solvated trialanine only
- Timestep: `0.5 fs`
- Temperature: `293 K`
- Pressure: `1 bar`
- NVT steps: `2000`, total `1 ps`
- NPT steps: `2000`, total `1 ps`
- Node type: `gpusmall`
- Array layout: task `0=off`, `1=omol`, `2=mh1`, `3=polar`

Benchmark outputs:
- `outputs/benchmark_solution_<model>_job<jobid>_task<taskid>.nvt.log`
- `outputs/benchmark_solution_<model>_job<jobid>_task<taskid>.nvt.traj`
- `outputs/benchmark_solution_<model>_job<jobid>_task<taskid>.npt.log`
- `outputs/benchmark_solution_<model>_job<jobid>_task<taskid>.npt.traj`
- `structures/benchmark_trialanine_solution_<model>_nvt.extxyz`
- `structures/benchmark_trialanine_solution_<model>_npt.extxyz`

May 1, 2026 `polar + cueq` benchmark on `gpubig`:
- Helper script: `/fhi/home/ngoen/software/venv_backups/triala_polar_cueq_bench_2026-05-01.py`
- Outputs:
  - `outputs/polar_cueq_benchmark_2026-05-01.nvt.log`
  - `outputs/polar_cueq_benchmark_2026-05-01.nvt.traj`
  - `outputs/polar_cueq_benchmark_2026-05-01.nvt.extxyz`
  - `outputs/polar_cueq_benchmark_2026-05-01.npt.log`
  - `outputs/polar_cueq_benchmark_2026-05-01.npt.traj`
  - `outputs/polar_cueq_benchmark_2026-05-01.npt.extxyz`
- Timing:
  - `1 ps` NVT: `1517.63 s` = `25.29 min`
  - `1 ps` NPT: `2792.05 s` = `46.53 min`
  - Extrapolated `1 ns` NVT: `421.56 h/ns` = `17.57 d/ns`
  - Extrapolated `1 ns` NPT: `775.57 h/ns` = `32.32 d/ns`

May 2, 2026 `polar + cueq` benchmark for `medium` and `small` on `gpubig`:
- Helper script: `/fhi/home/ngoen/software/venv_backups/triala_polar_cueq_bench_param_2026-05-02.py`
- `medium` outputs:
  - `outputs/polar_cueq_benchmark_medium_2026-05-02.nvt.log`
  - `outputs/polar_cueq_benchmark_medium_2026-05-02.nvt.traj`
  - `outputs/polar_cueq_benchmark_medium_2026-05-02.nvt.extxyz`
  - `outputs/polar_cueq_benchmark_medium_2026-05-02.npt.log`
  - `outputs/polar_cueq_benchmark_medium_2026-05-02.npt.traj`
  - `outputs/polar_cueq_benchmark_medium_2026-05-02.npt.extxyz`
- `medium` timing:
  - `1 ps` NVT: `1066.38 s` = `17.77 min`
  - `1 ps` NPT: `2777.65 s` = `46.29 min`
  - Extrapolated `1 ns` NVT: `296.22 h/ns` = `12.34 d/ns`
  - Extrapolated `1 ns` NPT: `771.57 h/ns` = `32.15 d/ns`
- `small` outputs:
  - `outputs/polar_cueq_benchmark_small_2026-05-02.nvt.log`
  - `outputs/polar_cueq_benchmark_small_2026-05-02.nvt.traj`
  - `outputs/polar_cueq_benchmark_small_2026-05-02.nvt.extxyz`
  - `outputs/polar_cueq_benchmark_small_2026-05-02.npt.log`
  - `outputs/polar_cueq_benchmark_small_2026-05-02.npt.traj`
  - `outputs/polar_cueq_benchmark_small_2026-05-02.npt.extxyz`
- `small` timing:
  - `1 ps` NVT: `1299.96 s` = `21.67 min`
  - `1 ps` NPT: `1732.22 s` = `28.87 min`
  - Extrapolated `1 ns` NVT: `361.10 h/ns` = `15.05 d/ns`
  - Extrapolated `1 ns` NPT: `481.17 h/ns` = `20.05 d/ns`

May 4, 2026 `polar + cueq` medium-model optimization and equilibration:
- Switched the shared `polar` model entry in `scripts/run_mace_trialanine.py` from `polar-1-l` to `polar-1-m` while keeping `enable_cueq=True`.
- Fresh medium-model minimization output:
  - `outputs/solution_polar_medium_cueq_opt_2026-05-04.bfgs.log`
  - `outputs/solution_polar_medium_cueq_opt_2026-05-04.extxyz`
- Submitted full medium-model solution equilibration on `fhi-raccoon`:
  - Slurm job `43210`
  - Job name: `triala_sol_polar_m`
  - Partition: `gpubig`
  - One GPU, one task, `16` CPUs, `64 GB` memory
  - Includes minimization, `20 ps` NVT, and `100 ps` NPT
- Target outputs for the running equilibration:
  - `outputs/solution_polar_medium_cueq_eq_2026-05-04.nvt.log`
  - `outputs/solution_polar_medium_cueq_eq_2026-05-04.npt.log`
  - `structures/trialanine_solution_polar_medium_nvt.extxyz`
  - `structures/trialanine_solution_polar_medium_npt.extxyz`

## 6. Viper Polar Restart

Viper project path:
- `/ptmp/ngoen/Documents/Enhanced_sampling/MACE/trialanine_mace_metad`

Purpose:
- Continue the interrupted solvated `polar` NPT on `viper-gpu` after repeated `fhi-raccoon` GPU memory issues on `gpusmall`.

Copied restart inputs:
- `structures/trialanine_solution_polar_nvt.extxyz`
- `outputs/solution_polar_eq_job35684_task1_gpu1.npt.traj`
- `models/MACEPOLAR1Lmodel`

Viper restart files:
- `README.md`
- `config.env`
- `run_polar_npt_continue.py`
- `run_polar_npt_continue_viper.sh`
- `submit_polar_npt_continue.sh`

Viper environment notes:
- Base venv: `/ptmp/ngoen/mace_upt_venv`
- Modules: `gcc/14`, `rocm/6.4`, `python-waterboa/2024.06`
- The original viper MACE install was missing `mace_polar`, `PolarMACE`, and related polar dependencies.
- To align behavior with `fhi-raccoon`, the viper venv was updated in place by copying the `mace` package, `mace/calculators`, `mace/modules`, and `graph_longrange` from `/fhi/home/ngoen/software/biASE-venv`.

Viper restart constants:
- remaining NPT steps: `194700`
- timestep: `0.5 fs`
- temperature: `293 K`
- pressure: `1 bar`
- `taut = 1 ps`
- `taup = 2 ps`
- default dtype: `float32`

Submission:
- Viper job `8197641`
- Job name: `triala_polar_viper_npt`
- Submitted on `apu`
- Walltime: `23:59:59` because the viper `apu` queue rejected longer runtimes
