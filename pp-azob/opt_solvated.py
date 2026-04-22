from mace.calculators import mace_off
from ase.io import read,write
from ase.optimize import BFGS,FIRE


mace_calc = mace_off(model="medium",device="cuda:0",default_dtype="float64",dispersion=False)

atoms = read("opt_solvated_pp-azob_trans_dmso.xyz")
atoms.calc=mace_calc
BFGS(atoms).run(fmax=0.05)
write("opt_solvated_pp-azob_trans_dmso.xyz",atoms)