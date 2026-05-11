from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.optimize import BFGS

from so3lr import So3lrCalculator


START_FILES = {
    "cis": "azobenzene_cis.xyz",
    "trans": "azobenzene_trans.xyz",
}


def build_calculator():
    return So3lrCalculator(calculate_stress=False, lr_cutoff=1000, dtype=np.float64)


def optimize_system(label: str, start_file: str):
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)

    atoms = read(start_file)
    atoms.calc = build_calculator()

    traj_path = outputs_dir / f"azob_{label}_opt.traj"
    log_path = outputs_dir / f"bfgs_azob_{label}.log"
    xyz_path = outputs_dir / f"azob_{label}_opt.xyz"

    optimizer = BFGS(atoms, trajectory=str(traj_path), logfile=str(log_path))
    optimizer.run(fmax=0.05, steps=500)

    write(xyz_path, atoms)
    energy = atoms.get_potential_energy()
    print(f"{label}: saved {xyz_path} with energy {energy:.10f} eV")


def main():
    for label, start_file in START_FILES.items():
        optimize_system(label, start_file)


if __name__ == "__main__":
    main()
