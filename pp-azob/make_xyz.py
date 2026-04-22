from ase.io import read

# Load the SDF file
atoms = read('azobenzene.sdf')

# Inspect the structure
print(atoms)
print(atoms.positions)  # Atomic positions
from ase.io import write

# Save to XYZ format
write('azobenzene.xyz', atoms)

