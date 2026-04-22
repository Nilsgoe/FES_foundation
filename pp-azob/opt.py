from mace.calculators import mace_off
from ase.io import read,write
from ase.optimize import BFGS,FIRE


mace_calc = mace_off(model="large",device="cuda:0",default_dtype="float64",dispersion=True)
'''atoms = read("pp-azobenzene_cis.xyz")
atoms.calc=mace_calc
FIRE(atoms).run(fmax=0.05)
write("pp-azob_cis_opt.traj",atoms)

atoms = read("pp-azobenzene_trans.xyz")
atoms.calc=mace_calc
FIRE(atoms).run(fmax=0.05)
write("pp-azob_trans.traj",atoms)
'''

atoms = read("dmso.xyz")
atoms.calc=mace_calcmv op
BFGS(atoms).run(fmax=0.05)
write("dmso_opt.xyz",atoms)