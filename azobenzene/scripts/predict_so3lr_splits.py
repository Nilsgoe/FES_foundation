#!/usr/bin/env python3
"""Evaluate fine-tuned SO3LR on one azobenzene dataset split."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.experimental.layout as _jl
import numpy as np

if not hasattr(_jl, "DeviceLocalLayout"):
    _jl.DeviceLocalLayout = _jl.Layout

DATA_ROOT = Path("/ptmp/ngoen/Documents/azobenzene_so3lr_training")
FT_WORKDIR = DATA_ROOT / "ft_so3lr"
OUT_ROOT = Path(
    "/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/"
    "azobenzene_dft_benchmark/results/split_predictions"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "valid", "test"), required=True)
    return parser.parse_args()


def load_calc():
    from mlff.md import mlffCalculatorSparse

    # So3lrCalculator loads bundled pretrained parameters. This is the same
    # direct orbax checkpoint construction used by the working FT-SO3LR MetaD.
    return mlffCalculatorSparse.create_from_ckpt_dir(
        ckpt_dir=str(FT_WORKDIR),
        from_file=False,
        calculate_stress=False,
        lr_cutoff=1000,
        lr_neighbors_bool=True,
        dispersion_energy_cutoff_lr_damping=2.0,
    )


def main() -> None:
    from ase.io import read

    args = parse_args()
    input_xyz = DATA_ROOT / "data" / f"{args.split}_azob_so3lr.xyz"
    out_dir = OUT_ROOT / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = read(str(input_xyz), index=":")
    print(f"Split: {args.split}; frames: {len(frames)}")
    e_dft = np.array([atoms.get_potential_energy() for atoms in frames])
    f_dft = np.concatenate([atoms.get_forces() for atoms in frames], axis=0)
    n_atoms = np.array([len(atoms) for atoms in frames])

    calc = load_calc()
    e_pred, f_parts = [], []
    for index, atoms in enumerate(frames, start=1):
        evaluated = atoms.copy()
        evaluated.calc = calc
        e_pred.append(evaluated.get_potential_energy())
        f_parts.append(evaluated.get_forces())
        if index % 100 == 0:
            print(f"{index}/{len(frames)}")

    e_pred = np.array(e_pred)
    f_pred = np.concatenate(f_parts, axis=0)
    e_dft_pa = e_dft / n_atoms
    e_pred_pa = e_pred / n_atoms
    offset = float(np.mean(e_dft_pa - e_pred_pa))
    rmse_e = float(np.sqrt(np.mean((e_dft_pa - e_pred_pa - offset) ** 2)) * 1000)
    rmse_f = float(np.sqrt(np.mean((f_dft - f_pred) ** 2)))

    output = out_dir / "predictions_so3lr_ft.npz"
    np.savez(
        output,
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
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
