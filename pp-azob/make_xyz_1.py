from ase.io import read,write


atoms = read("pp-azob_trans.traj")

write("pp-azob_trans.xyz",atoms)

