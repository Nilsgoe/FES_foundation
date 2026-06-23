#!/usr/bin/env python3
"""Evaluate all trained MACE models on one azobenzene dataset split."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.io import read

BENCH_ROOT = Path("/nexus/posix0/FHI-Theory/ngoen/Enhanced_sampling/MACE/azobenzene_dft_benchmark")
OUT_ROOT = BENCH_ROOT / "results/split_predictions"
VIPER_DATA_ROOT = Path("/ptmp/ngoen/Documents/azobenzene_mace_training/data")

MODELS: dict[str, Path] = {
    "scratch": BENCH_ROOT / "models/azob_scratch_stagetwo.model",
    "ft_off24": BENCH_ROOT / "models/azob_ft_off.model",
    "ft_mh1": BENCH_ROOT / "models/azob_ft_mh1.model",
    "ft_mh1_avg_e0s": BENCH_ROOT / "models/azob_ft_mh1_avg_e0s.model",
}


def evaluate_model(model_path: Path, frames: list) -> tuple[np.ndarray, np.ndarray]:
    from mace.calculators import MACECalculator

    calc = MACECalculator(
        model_paths=[str(model_path)],
        device="cuda",
        default_dtype="float64",
        enable_cueq=False,
    )
    e_pred, f_parts = [], []
    for atoms in frames:
        a = atoms.copy()
        a.calc = calc
        e_pred.append(a.get_potential_energy())
        f_parts.append(a.get_forces())
    return np.array(e_pred), np.concatenate(f_parts, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.split == "train" and (VIPER_DATA_ROOT / "train_azob_noatoms.xyz").exists():
        input_xyz = VIPER_DATA_ROOT / "train_azob_noatoms.xyz"
    else:
        input_xyz = BENCH_ROOT / f"{args.split}_azob.xyz"
    out_dir = OUT_ROOT / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = read(str(input_xyz), index=":")
    print(f"Split: {args.split}; frames: {len(frames)}")

    e_dft = np.array([a.get_potential_energy() for a in frames])
    f_dft_parts = [atoms.get_forces() for atoms in frames]
    f_dft = np.concatenate(f_dft_parts, axis=0)
    n_atoms = np.array([len(a) for a in frames])

    np.savez(out_dir / "dft_reference.npz", e=e_dft, forces=f_dft, n_atoms=n_atoms)
    print("Saved dft_reference.npz")

    for name, model_path in MODELS.items():
        if not model_path.exists():
            print(f"SKIP {name}: {model_path} not found")
            continue
        print(f"\nEvaluating {name} …")
        e_pred, f_pred = evaluate_model(model_path, frames)

        e_dft_pa = e_dft / n_atoms
        e_pred_pa = e_pred / n_atoms
        delta = float(np.mean(e_dft_pa - e_pred_pa))
        rmse_e = float(np.sqrt(np.mean((e_dft_pa - e_pred_pa - delta) ** 2)) * 1000)  # meV/atom
        rmse_f = float(np.sqrt(np.mean((f_dft - f_pred) ** 2)))  # eV/Å

        out_path = out_dir / f"predictions_{name}.npz"
        np.savez(
            out_path,
            e=e_pred,
            forces=f_pred,
            n_atoms=n_atoms,
            rmse_e_mev=rmse_e,
            rmse_f_evang=rmse_f,
        )
        print(f"  RMSE_E = {rmse_e:.2f} meV/atom   RMSE_F = {rmse_f:.4f} eV/Å")
        print(f"  Saved {out_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
