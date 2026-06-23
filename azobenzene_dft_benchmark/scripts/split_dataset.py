"""
80/10/10 train/valid/test split of azobenzene_dft.xyz.
Atomic reference frames (H, C, N) are appended to the train file only.

Usage
-----
  python split_dataset.py \
      --dataset azobenzene_dft.xyz \
      --atomic-base /viper/ptmp1/ngoen/Documents/azobenzene_dft/atomic_energies \
      --outdir .
"""

import argparse
import random
from pathlib import Path

from ase import Atoms
from ase.io import read, write

EV_PER_EH = 27.211386246


def read_atomic_frame(atom_dir: Path, symbol: str) -> Atoms:
    out = atom_dir / symbol / "output.out"
    energy_eh = None
    for line in out.read_text().splitlines():
        if "FINAL SINGLE POINT ENERGY" in line:
            energy_eh = float(line.split()[-1])
    if energy_eh is None:
        raise ValueError(f"No energy in {out}")
    atoms = Atoms(symbol, positions=[[0.0, 0.0, 0.0]])
    atoms.info["energy"] = energy_eh * EV_PER_EH
    atoms.info["free_energy"] = energy_eh * EV_PER_EH
    atoms.info["config_type"] = f"atom_{symbol}"
    return atoms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--atomic-base", required=True)
    parser.add_argument("--outdir", default=".")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    frames = read(args.dataset, index=":")
    n = len(frames)

    rng = random.Random(args.seed)
    indices = list(range(n))
    rng.shuffle(indices)

    n_test  = n // 10           # 240
    n_valid = n // 10           # 240
    n_train = n - n_test - n_valid  # 1920

    test_idx  = sorted(indices[:n_test])
    valid_idx = sorted(indices[n_test: n_test + n_valid])
    train_idx = sorted(indices[n_test + n_valid:])

    train = [frames[i] for i in train_idx]
    valid = [frames[i] for i in valid_idx]
    test  = [frames[i] for i in test_idx]

    # Append atomic frames to train only
    atomic_base = Path(args.atomic_base)
    for sym in ("H", "C", "N"):
        train.append(read_atomic_frame(atomic_base, sym))

    write(str(outdir / "train_azob.xyz"), train, format="extxyz")
    write(str(outdir / "valid_azob.xyz"), valid, format="extxyz")
    write(str(outdir / "test_azob.xyz"),  test,  format="extxyz")

    print(f"train: {len(train)} frames ({n_train} molecular + 3 atomic)")
    print(f"valid: {len(valid)} frames")
    print(f"test:  {len(test)} frames")
    print(f"Wrote to {outdir}/")


if __name__ == "__main__":
    main()
