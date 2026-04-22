import matplotlib.pyplot as plt
from ase.io import read
from ase.visualize.plot import plot_atoms

def save_traj_as_png(traj_file1, traj_file2, output1, output2, dpi=400):
    """
    Visualize and save the first frame of two trajectory files as PNG images.

    Parameters:
        traj_file1 (str): Path to the first trajectory file.
        traj_file2 (str): Path to the second trajectory file.
        output1 (str): Output path for the first PNG file.
        output2 (str): Output path for the second PNG file.
        dpi (int): Resolution of the PNG images (default: 400).
    """
    # Read the first frame from each trajectory file
    atoms1 = read(traj_file1, index=0)
    atoms2 = read(traj_file2, index=0)
    
    # Visualize and save the first trajectory
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_atoms(atoms1, ax=ax, radii=0.9, rotation=('90x, 0y, 0z'))  # Adjust radii/rotation if needed
    ax.axis('off')  # Hide axes
    plt.savefig(output1, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    # Visualize and save the second trajectory
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_atoms(atoms2, ax=ax, radii=0.9, rotation=('90x, 0y, 180z'))  # Adjust radii/rotation if needed
    ax.axis('off')  # Hide axes
    plt.savefig(output2, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"Images saved as {output1} and {output2} at {dpi} DPI!")

# Example usage
save_traj_as_png("azob_trans_opt.traj", "azob_cis_opt.traj", "azob_trans.png", "azob_cis.png")

