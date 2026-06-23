from __future__ import annotations

import argparse
import csv
import os
import shutil
import time
from pathlib import Path

import numpy as np
from ase.calculators.orca import ORCA, OrcaProfile
from ase.io import read


ORCA_BIN = os.environ.get("ORCA_BIN", "/mpcdf/soft/RHEL_9/packages/x86_64/orca/6.1.1/bin/orca")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASE/ORCA force-evaluation benchmark for malonaldehyde.")
    parser.add_argument("--cores", type=int, required=True, help="ORCA PAL nprocs value.")
    parser.add_argument("--repeat", type=int, default=1, help="Number of independent force evaluations.")
    parser.add_argument("--xyz", type=Path, default=Path("malonaldehyde.xyz"))
    parser.add_argument("--out", type=Path, default=Path("timings.csv"))
    parser.add_argument("--work-root", type=Path, default=Path("orca_runs"))
    return parser.parse_args()


def append_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    atoms = read(args.xyz)
    args.work_root.mkdir(parents=True, exist_ok=True)

    for repeat_idx in range(args.repeat):
        run_dir = args.work_root / f"cores_{args.cores:02d}_repeat_{repeat_idx:02d}"
        if run_dir.exists():
            shutil.rmtree(run_dir)

        calc = ORCA(
            profile=OrcaProfile(command=ORCA_BIN),
            directory=run_dir,
            charge=0,
            mult=1,
            orcasimpleinput="wB97M-V def2-TZVPD EnGrad",
            orcablocks=f"%pal nprocs {args.cores} end",
        )
        atoms.calc = calc

        start = time.perf_counter()
        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        elapsed_s = time.perf_counter() - start

        row = {
            "cores": args.cores,
            "repeat": repeat_idx,
            "elapsed_s": f"{elapsed_s:.6f}",
            "energy_eV": f"{energy:.12f}",
            "max_force_eV_A": f"{float(np.max(np.linalg.norm(forces, axis=1))):.8f}",
            "natoms": len(atoms),
            "orca_bin": ORCA_BIN,
            "method": "wB97M-V/def2-TZVPD EnGrad",
            "work_dir": str(run_dir),
        }
        append_row(args.out, row)
        print(row, flush=True)


if __name__ == "__main__":
    main()
