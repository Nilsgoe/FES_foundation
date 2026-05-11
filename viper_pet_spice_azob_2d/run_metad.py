import argparse
from pathlib import Path

import jax.numpy as jnp
from ase import units
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS
from upet.calculator import UPETCalculator

from Metadynamics import WT_Metadynamics


TEMPERATURE_K = 333.0
FRICTION = 0.1
TIMESTEP = 0.5 * units.fs
BIAS_HEIGHT = 0.1
BIAS_FACTOR = 10
INTERVAL_SIZE = 100
MAX_BIAS = 200000
STD_DEV_2D = [5.0, 5.0]
DIHEDRAL_BOUNDS = (-180.0, 180.0)
ANGLE_BOUNDS = (0.0, 180.0)

INDEX_MAP = {
    "cis": {
        "dihedral": (1, 6, 7, 8),
        "angle": (1, 6, 7),
        "start_file": "azob_cis_opt.traj",
    },
    "trans": {
        "dihedral": (2, 11, 12, 13),
        "angle": (2, 11, 12),
        "start_file": "azob_trans_opt.traj",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Restartable azobenzene 2D PET-SPICE metadynamics.")
    parser.add_argument("--system", choices=["cis", "trans"], required=True)
    parser.add_argument("--chunk-id", type=int, required=True)
    parser.add_argument("--steps-per-chunk", type=int, required=True)
    parser.add_argument("--start-file", default=None)
    return parser.parse_args()


def build_calculator():
    return UPETCalculator(model="pet-spice-l", device="cuda")


def compute_dihedral(positions, indices):
    p1, p2, p3, p4 = positions[jnp.array(indices)]
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    n1 = jnp.cross(b1, b2)
    n2 = jnp.cross(b2, b3)
    n1 = n1 / jnp.linalg.norm(n1)
    n2 = n2 / jnp.linalg.norm(n2)
    b2_unit = b2 / jnp.linalg.norm(b2)

    cos_phi = jnp.dot(n1, n2)
    sin_phi = jnp.dot(jnp.cross(n1, n2), b2_unit)
    return jnp.degrees(jnp.arctan2(sin_phi, cos_phi))


def compute_angle(positions, indices):
    pa, pb, pc = positions[jnp.array(indices)]
    v1 = pa - pb
    v2 = pc - pb
    v1_u = v1 / jnp.linalg.norm(v1)
    v2_u = v2 / jnp.linalg.norm(v2)
    cos_theta = jnp.clip(jnp.dot(v1_u, v2_u), -1.0, 1.0)
    return jnp.degrees(jnp.arccos(cos_theta))


def build_cv(system):
    dihedral_indices = INDEX_MAP[system]["dihedral"]
    angle_indices = INDEX_MAP[system]["angle"]

    def cv_2d(positions):
        return jnp.array(
            [
                compute_dihedral(positions, dihedral_indices),
                compute_angle(positions, angle_indices),
            ]
        )

    return cv_2d


def build_paths(system):
    run_tag = f"pet_spice_azob_{system}_2d"
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    return (
        run_tag,
        outputs_dir / f"{run_tag}.traj",
        outputs_dir / f"{run_tag}.bias",
        outputs_dir / f"bfgs_{run_tag}.log",
        outputs_dir / f"{run_tag}_chunk_last.xyz",
    )


def main():
    args = parse_args()
    start_file = args.start_file or INDEX_MAP[args.system]["start_file"]
    run_tag, trajectory_path, bias_path, bfgs_log, final_xyz = build_paths(args.system)

    continue_run = args.chunk_id > 0 and trajectory_path.exists() and bias_path.exists()
    if args.chunk_id > 0 and not continue_run:
        raise FileNotFoundError(
            f"Missing restart files for chunk {args.chunk_id}: "
            f"{trajectory_path} and/or {bias_path}"
        )

    if continue_run:
        atoms = read(f"{trajectory_path}@-1").copy()
    else:
        atoms = read(start_file).copy()

    atoms.calc = build_calculator()
    if not continue_run:
        BFGS(atoms, logfile=str(bfgs_log)).run(fmax=0.05, steps=500)
        MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE_K)

    dyn = WT_Metadynamics(
        atoms=atoms,
        timestep=TIMESTEP,
        temperature_K=TEMPERATURE_K,
        friction=FRICTION,
        trajectory=str(trajectory_path),
        fixcm=True,
        cvs=build_cv(args.system),
        bias_height=BIAS_HEIGHT,
        interval_size=INTERVAL_SIZE,
        output_file=str(bias_path),
        well_temp=True,
        bias_factor=BIAS_FACTOR,
        append_trajectory=continue_run,
        input_file=str(bias_path) if continue_run else None,
        max_bias=MAX_BIAS,
        std_dev=STD_DEV_2D,
        wrapping=[True, False],
        bounds=(DIHEDRAL_BOUNDS, ANGLE_BOUNDS),
    )
    dyn.run(args.steps_per_chunk)

    write(final_xyz, atoms)
    print(f"Completed {run_tag} chunk {args.chunk_id} for {args.steps_per_chunk} steps")


if __name__ == "__main__":
    main()
