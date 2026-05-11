## Viper Polar NPT Restart

This bundle migrates the interrupted solvated trialanine `polar` NPT continuation from `fhi-raccoon` to `viper-gpu`.

Source project:
- `/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/trialanine_mace_metad`

Target project on viper:
- `/ptmp/ngoen/Documents/Enhanced_sampling/MACE/trialanine_mace_metad`

Restart inputs copied to viper:
- `structures/trialanine_solution_polar_nvt.extxyz`
- `outputs/solution_polar_eq_job35684_task1_gpu1.npt.traj`
- `models/MACEPOLAR1Lmodel`

The viper-side restart uses:
- `0.5 fs`
- `293 K`
- `1 bar`
- `taut = 1 ps`
- `taup = 2 ps`
- remaining NPT steps: `194700`

The calculator is instantiated directly with:
- `MACECalculator(model_paths="models/MACEPOLAR1Lmodel", model_type="PolarMACE", default_dtype="float32", device="cuda")`

This avoids relying on `mace.calculators.mace_polar`, which is not exposed by the current viper MACE environment.
