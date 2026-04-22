from mace.calculators import mace_off
from ase.io import read,write,Trajectory
from ase.md import Langevin
from ase import units

mace_calc = mace_off(model="large",device="cuda:0",default_dtype="float64",dispersion=False,enable_cueq=True)

atoms = read("opt_solvated_dmso.xyz")
atoms.calc=mace_calc
md = Langevin(atoms,timestep= .5 * units.fs, temperature=300 * units.kB, friction=0.01, )

traj = Trajectory('md_cu_eq_dmso.traj', 'w', atoms)
md.attach(traj.write, interval=10)
md.run(2e5)
