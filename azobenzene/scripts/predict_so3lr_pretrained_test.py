#!/usr/bin/env python3
"""Evaluate pretrained SO3LR on the azobenzene test split."""

from __future__ import annotations

from pathlib import Path

import jax.experimental.layout as _jl
import numpy as np

if not hasattr(_jl, "DeviceLocalLayout"):
    _jl.DeviceLocalLayout = _jl.Layout

DATA_ROOT = Path("/ptmp/ngoen/Documents/azobenzene_so3lr_training")
INPUT_XYZ = DATA_ROOT / "data/test_azob_so3lr.xyz"
OUTPUT = Path(
    "/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/"
    "azobenzene_dft_benchmark/results/split_predictions/test/"
    "predictions_so3lr_pretrained.npz"
)


def main() -> None:
    from ase.io import read
    from so3lr import So3lrCalculator

    frames = read(str(INPUT_XYZ), index=":")
    print(f"Test frames: {len(frames)}")
    e_dft = np.array([atoms.get_potential_energy() for atoms in frames])
    f_dft = np.concatenate([atoms.get_forces() for atoms in frames], axis=0)
    n_atoms = np.array([len(atoms) for atoms in frames])

    calc = So3lrCalculator(calculate_stress=False, lr_cutoff=1000, dtype=np.float64)
    e_pred, f_parts = [], []
    for index, atoms in enumerate(frames, start=1):
        evaluated = atoms.copy()
        evaluated.calc = calc
        e_pred.append(evaluated.get_potential_energy())
        f_parts.append(evaluated.get_forces())
        if index % 50 == 0:
            print(f"{index}/{len(frames)}")

    e_pred = np.array(e_pred)
    f_pred = np.concatenate(f_parts, axis=0)
    e_dft_pa = e_dft / n_atoms
    e_pred_pa = e_pred / n_atoms
    offset = float(np.mean(e_dft_pa - e_pred_pa))
    rmse_e = float(np.sqrt(np.mean((e_dft_pa - e_pred_pa - offset) ** 2)) * 1000)
    rmse_f = float(np.sqrt(np.mean((f_dft - f_pred) ** 2)))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT,
        e=e_pred,
        e_dft=e_dft,
        forces=f_pred,
        f_dft=f_dft,
        n_atoms=n_atoms,
        rmse_e_mev=rmse_e,
        rmse_f_evang=rmse_f,
        energy_reference="e0s_and_mean_subtracted",
    )
    print(f"RMSE_E={rmse_e:.3f} meV/atom RMSE_F={rmse_f:.6f} eV/A")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
