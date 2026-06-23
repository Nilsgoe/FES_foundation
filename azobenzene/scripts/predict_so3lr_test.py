#!/usr/bin/env python3
"""Evaluate fine-tuned SO3LR on test_azob_so3lr.xyz, save predictions as NPZ.

Run on viper-gpu with the biase_venv_sol3r environment active.
E0s and mean shift are already applied in the test file — model output
is directly comparable to the stored DFT energies.
"""

from __future__ import annotations

import sys
import numpy as np
from pathlib import Path

# Patch for orbax compatibility with jax >= 0.6
import jax.experimental.layout as _jl
if not hasattr(_jl, "DeviceLocalLayout"):
    _jl.DeviceLocalLayout = _jl.Layout

DATA_ROOT = Path("/ptmp/ngoen/Documents/azobenzene_so3lr_training")
TEST_XYZ = DATA_ROOT / "data/test_azob_so3lr.xyz"
FT_WORKDIR = DATA_ROOT / "ft_so3lr"
OUT_DIR = Path("/ptmp/ngoen/Documents/azobenzene_mace_training/results/test_predictions")


def load_so3lr_calc():
    """Load fine-tuned SO3LR calculator from workdir."""
    try:
        from so3lr import So3lrCalculator
        calc = So3lrCalculator(workdir=str(FT_WORKDIR))
        return calc
    except TypeError:
        # Older API may not accept workdir positionally
        pass

    try:
        from so3lr import So3lrCalculator
        calc = So3lrCalculator(model_name="So3lr", workdir=str(FT_WORKDIR))
        return calc
    except TypeError:
        pass

    raise RuntimeError(
        f"Could not construct So3lrCalculator with workdir={FT_WORKDIR}.\n"
        "Check the so3lr version's Calculator API."
    )


def main() -> None:
    from ase.io import read

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading test set: {TEST_XYZ}")
    frames = read(str(TEST_XYZ), index=":")
    print(f"Test frames: {len(frames)}")

    e_dft = np.array([a.get_potential_energy() for a in frames])
    f_dft_parts = [a.arrays.get("forces", np.zeros((len(a), 3))) for a in frames]
    f_dft = np.concatenate(f_dft_parts, axis=0)
    n_atoms = np.array([len(a) for a in frames])

    print("Loading fine-tuned SO3LR model …")
    calc = load_so3lr_calc()

    e_pred, f_parts = [], []
    for i, atoms in enumerate(frames):
        a = atoms.copy()
        a.calc = calc
        e_pred.append(a.get_potential_energy())
        f_parts.append(a.get_forces())
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(frames)}")

    e_pred = np.array(e_pred)
    f_pred = np.concatenate(f_parts, axis=0)

    e_dft_pa = e_dft / n_atoms
    e_pred_pa = e_pred / n_atoms
    delta = float(np.mean(e_dft_pa - e_pred_pa))
    rmse_e = float(np.sqrt(np.mean((e_dft_pa - e_pred_pa - delta) ** 2)) * 1000)
    rmse_f = float(np.sqrt(np.mean((f_dft - f_pred) ** 2)))

    out_path = OUT_DIR / "predictions_so3lr_ft.npz"
    np.savez(
        out_path,
        e=e_pred,
        e_dft=e_dft,
        forces=f_pred,
        f_dft=f_dft,
        n_atoms=n_atoms,
        rmse_e_mev=rmse_e,
        rmse_f_evang=rmse_f,
        energy_reference="e0s_and_mean_subtracted",
    )
    print(f"\nRMSE_E = {rmse_e:.2f} meV/atom   RMSE_F = {rmse_f:.4f} eV/Å")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
