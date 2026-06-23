"""
Read atomic ORCA outputs for H, C, N and patch every frame in the extxyz
dataset with e0_H, e0_C, e0_N keys (values in eV).

Run after both extract_results.py and run_atomic_energies.slurm have finished.

Usage
-----
  python add_atomic_energies.py \
      --dataset azobenzene_dft.xyz \
      --atomic-base /viper/ptmp1/ngoen/Documents/azobenzene_dft/atomic_energies \
      --outfile azobenzene_dft_with_e0.xyz

  # Or supply values directly (eV) if you already have them:
  python add_atomic_energies.py \
      --dataset azobenzene_dft.xyz \
      --e0 H=-13.587 C=-1027.494 N=-1485.123 \
      --outfile azobenzene_dft_with_e0.xyz
"""

import argparse
from pathlib import Path

from ase.io import read, write

EV_PER_EH = 27.211386246


def read_atomic_energy(atom_dir: Path, symbol: str) -> float:
    out = atom_dir / symbol / "output.out"
    for line in reversed(out.read_text().splitlines()):
        if "FINAL SINGLE POINT ENERGY" in line:
            return float(line.split()[-1]) * EV_PER_EH
    raise ValueError(f"No energy found in {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Input extxyz file")
    parser.add_argument(
        "--atomic-base",
        default=None,
        help="Directory containing H/, C/, N/ ORCA output subdirs",
    )
    parser.add_argument(
        "--e0",
        nargs="+",
        metavar="SYM=VALUE",
        default=None,
        help="Atomic energies in eV, e.g. H=-13.587 C=-1027.494 N=-1485.123",
    )
    parser.add_argument("--outfile", required=True, help="Output extxyz file")
    args = parser.parse_args()

    if args.e0 is not None:
        e0 = {}
        for token in args.e0:
            sym, val = token.split("=")
            e0[sym.strip()] = float(val)
    elif args.atomic_base is not None:
        base = Path(args.atomic_base)
        e0 = {sym: read_atomic_energy(base, sym) for sym in ("H", "C", "N")}
    else:
        raise SystemExit("Provide either --atomic-base or --e0")

    print("Atomic reference energies (eV):")
    for sym, val in sorted(e0.items()):
        print(f"  {sym}: {val:.6f}")

    frames = read(args.dataset, index=":")
    for atoms in frames:
        for sym, val in e0.items():
            atoms.info[f"e0_{sym}"] = val

    write(args.outfile, frames, format="extxyz")
    print(f"\nWrote {len(frames)} frames to {args.outfile}")


if __name__ == "__main__":
    main()
