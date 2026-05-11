## Viper MH1 Solution MetaD

This bundle runs solvated trialanine `mh1` well-tempered MetaD on `viper-gpu`
using restartable day-sized chunks because the `apu` partition limits jobs to
`23:59:59`.

Source NPT structure from raccoon:
- `/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/trialanine_mace_metad/structures/trialanine_solution_mh1_npt.extxyz`

Target path on viper:
- `/ptmp/ngoen/Documents/Enhanced_sampling/MACE/trialanine_mace_metad_mh1`

Model / environment:
- `mace_mp(model="mh-1", head="omol", enable_cueq=False, device="cuda")`
- venv: `/ptmp/ngoen/mace_upt_venv`
- modules: `gcc/14`, `rocm/6.4`, `python-waterboa/2024.06`

MetaD settings:
- `0.5 fs`
- `293 K`
- `phi = (14, 16, 18, 24)`
- `psi = (16, 18, 24, 26)`
- bias height `0.1 eV`
- sigma `[5 deg, 5 deg]`
- pace `100`
- bias factor `10`

Default chunk settings:
- `20000` steps per chunk = `10 ps`
- `100` chunks = `1 ns`

The submit helper currently uses `afterok` so the chain stops if a chunk fails.
