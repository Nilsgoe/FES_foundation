import jax.numpy as jnp
from ase import units
from ase.calculators.mixing import SumCalculator
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS
from mace.calculators import mace_omol

from Metadynamics import WT_Metadynamics
from biASE import General_Bias_Calculator


def asymmetric_hookean_well(x, lower_boundary=-1.2, upper_boundary=1.2, k_lower=50, k_upper=50):
    below_lower = k_lower * jnp.square(lower_boundary - x)
    above_upper = k_upper * jnp.square(x - upper_boundary)
    inside_well = 0.0
    potential = jnp.where(
        x < lower_boundary,
        below_lower,
        jnp.where(x > upper_boundary, above_upper, inside_well),
    )
    return potential[0]


def compute_cv(positions):
    d_o4h8 = jnp.linalg.norm(positions[4] - positions[8])
    d_o3h8 = jnp.linalg.norm(positions[3] - positions[8])
    return jnp.array([d_o4h8 - d_o3h8])


def main():
    run_tag = 'omol_large'

    mace_calc = mace_omol(model='extra_large')
    atoms_initial = read('optimized_fmalonaldehyde_initial.xyz')
    well_calc = General_Bias_Calculator(
        cv_function=compute_cv,
        bias_function=asymmetric_hookean_well,
    )
    combined_calc = SumCalculator([mace_calc, well_calc])

    atoms = atoms_initial.copy()
    atoms.calc = combined_calc
    BFGS(atoms, logfile=f'./outputs/bfgs_{run_tag}.log').run(fmax=0.01, steps=500)

    MaxwellBoltzmannDistribution(atoms, temperature_K=293)
    dyn = WT_Metadynamics(
        atoms,
        timestep=0.5 * units.fs,
        temperature_K=293,
        friction=0.1,
        trajectory=f'./outputs/metad_{run_tag}.traj',
        fixcm=False,
        cvs=compute_cv,
        std_dev=0.02,
        bias_height=0.05,
        interval_size=100,
        output_file=f'./outputs/metad_{run_tag}.txt',
        bias_factor=5,
    )
    dyn.run(1e6)


if __name__ == '__main__':
    main()
