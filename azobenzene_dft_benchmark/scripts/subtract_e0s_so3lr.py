"""
Subtract atomic reference energies and mean training energy from DFT energies
to produce centered, relative energies suitable for so3lr fine-tuning.

Pipeline:
  1. Subtract atomic E0s (wB97M-V/def2-TZVPD) to get formation-like energies
  2. Subtract mean training energy to center around 0

The mean is computed from the training set only and applied consistently to
all splits. This avoids so3lr's shift_mode='mean' which has a jax type bug.

E0s (wB97M-V/def2-TZVPD, eV):
  H: -13.445423
  C: -1029.854265
  N: -1485.541877

Usage
-----
  python subtract_e0s_so3lr.py --indir <dir> --outdir <dir>
"""

import argparse
from pathlib import Path
from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator

E0S = {"H": -13.445423, "C": -1029.854265, "N": -1485.541877}


def subtract_e0s(frames: list) -> list:
    out = []
    for i, atoms in enumerate(frames):
        if atoms.calc is None:
            raise ValueError(f"Frame {i} ({atoms.get_chemical_formula()}) has no calculator/energy")
        e0_sum = sum(E0S[s] for s in atoms.get_chemical_symbols())
        results = dict(atoms.calc.results)  # read before copy() clears calc
        results["energy"] = float(results["energy"]) - e0_sum
        if "free_energy" in results:
            results["free_energy"] = float(results["free_energy"]) - e0_sum
        atoms = atoms.copy()
        atoms.calc = SinglePointCalculator(atoms, **results)
        out.append(atoms)
    return out


def shift_by_mean(frames: list, mean_energy: float) -> list:
    out = []
    for atoms in frames:
        results = dict(atoms.calc.results)
        results["energy"] = results["energy"] - mean_energy
        if "free_energy" in results:
            results["free_energy"] = results["free_energy"] - mean_energy
        atoms = atoms.copy()
        atoms.calc = SinglePointCalculator(atoms, **results)
        out.append(atoms)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": indir / "train_azob_noatoms.xyz",
        "valid": indir / "valid_azob.xyz",
        "test":  indir / "test_azob.xyz",
    }

    # Step 1: subtract E0s from all splits
    processed = {}
    for split, src in splits.items():
        frames = read(str(src), index=":")
        processed[split] = subtract_e0s(frames)

    # Step 2: compute mean energy from training set only, apply to all splits
    train_energies = [f.calc.results["energy"] for f in processed["train"]]
    mean_energy = sum(train_energies) / len(train_energies)
    print(f"Training mean energy (post-E0): {mean_energy:.4f} eV  →  subtracting from all splits")

    for split, frames in processed.items():
        frames = shift_by_mean(frames, mean_energy)
        processed[split] = frames
        e_vals = [f.calc.results["energy"] for f in frames]
        print(f"{split}: {len(frames)} frames, "
              f"energy range [{min(e_vals):.3f}, {max(e_vals):.3f}] eV")
        write(str(outdir / f"{split}_azob_so3lr.xyz"), frames, format="extxyz")

    # Combined train+valid for so3lr --datafile (train first, then valid)
    combined = processed["train"] + processed["valid"]
    write(str(outdir / "azob_so3lr.xyz"), combined, format="extxyz")
    print(f"combined (train+valid): {len(combined)} frames -> azob_so3lr.xyz")
    print(f"mean_energy_shift={mean_energy:.6f} eV  (add back to recover formation energies)")


if __name__ == "__main__":
    main()
