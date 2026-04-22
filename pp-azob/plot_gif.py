from ase.io import read
from ase.visualize import view, write

# Load the trajectory file
traj_file = "azob_dis_no_consbiased_8_large_new_cv.traj"
atoms_list = read(traj_file, index=":")  # Read all frames

# Output GIF filename
output_gif = "trajectory_visualization.gif"

# Create a GIF using ASE's write function
write(output_gif, atoms_list, format='gif', fps=10)  # fps=10 for 10 frames per second

print(f"GIF created: {output_gif}")
