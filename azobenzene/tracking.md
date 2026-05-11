# Azobenzene MetaD Tracking

Cluster: `fhi-raccoon`

Layout:
- `4` submitted SLURM jobs: `off`, `omol`, `mh1`, `polar`
- each submitted job follows the same `run_U_MDs_cueq.sh` style as `malonaldehyd`
- each submitted job uses `1` node with `2` GPU tasks
- SLURM array `0-1%1` maps to `cis` and `trans`
- inside each array task:
  `GPU0 -> 1d MetaD`
  `GPU1 -> 2d MetaD`

Important files:
- driver: `run_metad.py`
- raccoon launcher: `run_metad_raccoon.sh`
- submit helper: `submit_metad_raccoon.sh`

Venvs:
- default / legacy: `/fhi/home/ngoen/software/biASE-venv`
- cueq-tested replacement for `off` and `omol`: `/fhi/home/ngoen/software/mace_cueq_venv`

## Submitted Jobs

| Model | Job ID | Submit Date | State | Notes |
| --- | --- | --- | --- | --- |
| off | 33975 | 2026-04-21 | running/pending array | resubmitted with `/fhi/home/ngoen/software/mace_cueq_venv`; `run_metad.py` now lazily imports MACE calculators so off/omol do not fail on missing `mace_polar` |
| omol | 33976 | 2026-04-21 | pending | resubmitted with `/fhi/home/ngoen/software/mace_cueq_venv`; `run_metad.py` now lazily imports MACE calculators so off/omol do not fail on missing `mace_polar` |
| mh1 | | | | |
| polar | | | | |

## Per-Run Status

| Model | System | CV | Status | Outputs | Notes |
| --- | --- | --- | --- | --- | --- |
| off | cis | 1d | pending | | |
| off | cis | 2d | pending | | |
| off | trans | 1d | pending | | |
| off | trans | 2d | pending | | |
| omol | cis | 1d | pending | | |
| omol | cis | 2d | pending | | |
| omol | trans | 1d | pending | | |
| omol | trans | 2d | pending | | |
| mh1 | cis | 1d | pending | | |
| mh1 | cis | 2d | pending | | |
| mh1 | trans | 1d | pending | | |
| mh1 | trans | 2d | pending | | |
| polar | cis | 1d | pending | | |
| polar | cis | 2d | pending | | |
| polar | trans | 1d | pending | | |
| polar | trans | 2d | pending | | |

## Expected Output Prefixes

The Python runs write into `outputs/` with names like:
- `bfgs_azob_<system>_<model>_<cv>_<run_label>.log`
- `metad_azob_<system>_<model>_<cv>_<run_label>.traj`
- `metad_azob_<system>_<model>_<cv>_<run_label>.txt`

## Useful Commands

Submit all four model jobs:

```bash
./submit_metad_raccoon.sh
```

Check queue:

```bash
squeue -u ngoen
```

Check azobenzene logs:

```bash
ls -lt logs
```
