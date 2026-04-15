from ase import Atoms
from ase.io import read, write
from ase.optimize import BFGS
from mace.calculators import mace_off

# Load the malonaldehyde structure
structure = read("malon.xyz")

# Set up MACE calculator (update path to your model)
calculator = mace_off(model="large", dispersion=True)
structure.set_calculator(calculator)

# Run geometry optimization
optimizer = BFGS(structure)
optimizer.run(fmax=0.01)  # Convergence criterion (forces in eV/Å)

# Save the optimized structure
write("optimized_malonaldehyde_initial.xyz", structure)
