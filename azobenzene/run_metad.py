import argparse
from pathlib import Path

import jax.numpy as jnp
from ase import units
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS

from Metadynamics import WT_Metadynamics


SYSTEM_SPECS = {
    "cis": {
        "start_file": "azob_cis_opt.traj",
        "dihedral_indices": (1, 6, 7, 8),
        "angle_indices": (1, 6, 7),
    },
    "trans": {
        "start_file": "azob_trans_opt.traj",
        "dihedral_indices": (2, 11, 12, 13),
        "angle_indices": (2, 11, 12),
    },
}

MODEL_SPECS = {
    "off": {"family": "off", "size": "large"},
    "omol": {"family": "omol", "size": "extra_large"},
    "mh1": {"family": "mh1", "size": "mh-1"},
    "polar": {"family": "polar", "size": "l"},
}

METAD_SETTINGS = {
    "1d": {
        "std_dev": 5.0,
        "bias_height": 0.1,
        "interval_size": 100,
        "bias_factor": 10,
        "wrapping": [True],
        "bounds": ((-180.0, 180.0),),
    },
    "2d": {
        "std_dev": [5.0, 5.0],
        "bias_height": 0.1,
        "interval_size": 100,
        "bias_factor": 10,
        "wrapping": [True, False],
        "bounds": ((-180.0, 180.0), (0.0, 180.0)),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run azobenzene metadynamics with MACE models.")
    parser.add_argument("--system", choices=sorted(SYSTEM_SPECS), required=True)
    parser.add_argument("--model-key", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--cv-mode", choices=sorted(METAD_SETTINGS), required=True)
    parser.add_argument("--run-label", default="", help="Optional label appended to the output tag.")
    parser.add_argument("--steps", type=int, default=int(1e6), help="Number of MD steps.")
    parser.add_argument(
        "--continue-run",
        action="store_true",
        help="Restart from the last frame of an existing trajectory and append to the same bias/trajectory files.",
    )
    parser.add_argument(
        "--trajectory-loginterval",
        type=int,
        default=1,
        help="Write every Nth MD step to the trajectory.",
    )
    return parser.parse_args()


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
    phi = jnp.arctan2(sin_phi, cos_phi)
    return jnp.degrees(phi)


def compute_angle(positions, indices):
    pa, pb, pc = positions[jnp.array(indices)]
    v1 = pa - pb
    v2 = pc - pb
    v1_u = v1 / jnp.linalg.norm(v1)
    v2_u = v2 / jnp.linalg.norm(v2)
    cos_theta = jnp.clip(jnp.dot(v1_u, v2_u), -1.0, 1.0)
    return jnp.degrees(jnp.arccos(cos_theta))


def build_cv_function(system_name, cv_mode):
    spec = SYSTEM_SPECS[system_name]

    def cv_1d(positions):
        return jnp.array([compute_dihedral(positions, spec["dihedral_indices"])])

    def cv_2d(positions):
        return jnp.array(
            [
                compute_dihedral(positions, spec["dihedral_indices"]),
                compute_angle(positions, spec["angle_indices"]),
            ]
        )

    return cv_1d if cv_mode == "1d" else cv_2d


def build_calculator(model_key):
    spec = MODEL_SPECS[model_key]
    model_family = spec["family"]
    model_size = spec["size"]

    if model_family == "off":
        from mace.calculators import mace_off

        return mace_off(model=model_size, dispersion=True, enable_cueq=True)
    if model_family == "omol":
        from mace.calculators import mace_omol

        return mace_omol(model=model_size, enable_cueq=True)
    if model_family == "polar":
        try:
            from mace.calculators import mace_polar
        except ImportError as exc:
            raise ImportError(
                "mace_polar is unavailable in this environment. "
                "Use a MACE environment that provides mace.calculators.mace_polar "
                "for polar runs."
            ) from exc

        return mace_polar(model=f"polar-1-{model_size}", enable_cueq=False)
    if model_family == "mh1":
        from mace.calculators import mace_mp

        return mace_mp(model=model_size, head="omol", enable_cueq=True)
    raise ValueError(f"Unsupported model key: {model_key}")


def main():
    args = parse_args()
    system_spec = SYSTEM_SPECS[args.system]
    metad_spec = METAD_SETTINGS[args.cv_mode]

    run_tag = f"azob_{args.system}_{args.model_key}_{args.cv_mode}"
    if args.run_label:
        run_tag = f"{run_tag}_{args.run_label}"

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)

    trajectory_path = outputs_dir / f"metad_{run_tag}.traj"
    bias_path = outputs_dir / f"metad_{run_tag}.txt"

    if args.continue_run:
        if not trajectory_path.exists() or not bias_path.exists():
            raise FileNotFoundError(
                f"Continuation requested but missing {trajectory_path} and/or {bias_path}."
            )
        atoms = read(f"{trajectory_path}@-1").copy()
    else:
        atoms = read(system_spec["start_file"]).copy()

    if args.model_key == "polar" and not any(atoms.pbc) and atoms.cell.volume == 0:
        atoms.set_cell([50, 50, 50])
        atoms.center()

    atoms.calc = build_calculator(args.model_key)
    if not args.continue_run:
        BFGS(atoms, logfile=str(outputs_dir / f"bfgs_{run_tag}.log")).run(fmax=0.05, steps=500)
        MaxwellBoltzmannDistribution(atoms, temperature_K=333)

    dyn = WT_Metadynamics(
        atoms,
        timestep=0.5 * units.fs,
        temperature_K=333,
        friction=0.1,
        trajectory=str(trajectory_path),
        fixcm=False,
        cvs=build_cv_function(args.system, args.cv_mode),
        std_dev=metad_spec["std_dev"],
        bias_height=metad_spec["bias_height"],
        interval_size=metad_spec["interval_size"],
        output_file=str(bias_path),
        well_temp=True,
        bias_factor=metad_spec["bias_factor"],
        append_trajectory=args.continue_run,
        input_file=str(bias_path) if args.continue_run else None,
        wrapping=metad_spec["wrapping"],
        bounds=metad_spec["bounds"],
        max_bias=int(1e6),
        loginterval=args.trajectory_loginterval,
    )
    dyn.run(args.steps)


if __name__ == "__main__":
    main()
