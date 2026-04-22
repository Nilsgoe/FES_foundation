from biASE import General_Bias_Calculator
from ase.atoms import Atoms
import numpy as np
from ase.io import read,Trajectory
import csv
from mace.calculators import mace_off
import jax.numpy as jnp
from ase.optimize import BFGS
from ase.calculators.mixing import SumCalculator
from functools import partial
from ase.md.langevin import Langevin
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
import argparse

parser = argparse.ArgumentParser(description="Umbrella MD shifted from the a CV by input*0.05")
parser.add_argument('shift',type=int, help="Input value for the shift")
args = parser.parse_args()

def circular_mean(deg_angles):
    """
    Calculate the circular (angular) mean of a list of angles (in degrees)
    using the complex-exponential trick.
    """
    # Convert to radians
    rad_angles = deg_angles
    # Mean vector on the unit circle
    mean_vector = np.mean(np.exp(1j * rad_angles))
    return np.angle(mean_vector)


def compute_cv(positions):
    """
    Calculate the dihedral angle defined by four atomic positions.
    
    Parameters:
        positions (jnp.ndarray): A (4, 3) array containing the positions of the four atoms.
    
    Returns:
        float: Dihedral angle in degrees.
    """
    # Extract positions of the atoms
    p1, p2, p3, p4 = positions[jnp.array([2,11,12,13])]
    #print(p1,p2,p3,p4)
    # Define vectors
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    
    # Calculate normal vectors of the planes
    n1 = jnp.cross(b1, b2) 
    n2 = jnp.cross(b2, b3)
    
    # Normalize the normal vectors
    n1 /= jnp.linalg.norm(n1) + 1e-12
    n2 /= jnp.linalg.norm(n2) + 1e-12
    
    # Calculate unit vector along b2
    b2 /= jnp.linalg.norm(b2) + 1e-12
    
    # Calculate cosine and sine of the dihedral angle
    cos_phi = jnp.dot(n1, n2)
    sin_phi = jnp.dot(jnp.cross(n1, n2), b2)
    
    # Compute the angle in radians and convert to degrees
    phi = jnp.arctan2(sin_phi, cos_phi)
    return phi


def umbrella_potential(x, x0, k):
    """
    Compute a harmonic bias potential for a dihedral angle using JAX, handling boundary conditions smoothly.

    Parameters:
        phi (float): Current dihedral angle in degrees.
        phi_target (float): Target dihedral angle in degrees.
        k (float): Force constant.

    Returns:
        float: Harmonic bias potential.
    """
    # Wrap angles to [-180, 180]
    phi_wrapped = (x + 180) % 360 - 180
    phi_target_wrapped = (x0 + 180) % 360 - 180

    # Compute the potential with smooth boundary handling
    delta_phi = phi_wrapped - phi_target_wrapped
    #print(delta_phi)
    delta_phi = (delta_phi + 180) % 360 - 180 
    #print(delta_phi)
    # Compute harmonic potential
    bias = 0.5 * k * delta_phi**2
    return bias


def umbrella_potential_triangle(x, x0, k):
    """
    Compute a harmonic bias potential for a dihedral angle using trigonometric wrapping,
    ensuring smooth gradients for JAX automatic differentiation.

    Parameters:
        x (float): Current dihedral angle in degrees.
        x0 (float): Target dihedral angle in degrees.
        k (float): Force constant.

    Returns:
        float: Harmonic bias potential.
    """
    # Convert degrees to radians
    phi_rad = x
    phi_target_rad = x0

    # Compute the periodic difference using trigonometric wrapping
    delta_phi = jnp.arctan2(jnp.sin(phi_rad - phi_target_rad), jnp.cos(phi_rad - phi_target_rad))

    # Compute harmonic potential
    bias = 0.5 * k * delta_phi**2
    return bias


mace_calc_dis = mace_off(model="large",dispersion=True)#XTB(method="GFN2-xTB")#
atoms1 = Trajectory("azob_trans_all_no_cons_trig.traj")[5]
print(compute_cv(atoms1.positions))
x0=compute_cv(atoms1.positions)-0.1*args.shift
if x0>jnp.pi:
    x0-=2*jnp.pi
    print("Adjusted x0: ",x0)
elif x0<-jnp.pi:
    x0+=2*jnp.pi
    print("Adjusted x0: ",x0)
print(x0,.1*args.shift)
#exit()
k_=50.0
umbrella_bias_calc = General_Bias_Calculator(cv_function=compute_cv,bias_function=partial(umbrella_potential_triangle,x0=x0,k=k_))
biased_mace_calc=SumCalculator([mace_calc_dis,umbrella_bias_calc])
atoms = Trajectory("azob_trans_all_no_cons_trig.traj")[args.shift+5]
#print("Acctual cv:", compute_cv(atoms.positions))
atoms.calc = biased_mace_calc
#print("Forces",atoms.get_forces(),atoms.get_total_energy())
#exit()
BFGS(atoms).run(fmax=0.05,steps=500,)

print(x0)
#exit()
#MaxwellBoltzmannDistribution(atoms, temperature_K=293)
dyn = Langevin(atoms, timestep=.5 * units.fs, temperature_K=293, friction=0.1, trajectory=f'azob_dis_no_consbiased_{args.shift}_large_new_cv_trig_rad.traj',fixcm=False)#Andersen(atoms, timestep=.5 * units.fs, temperature_K=3000, andersen_prob=0.01,trajectory='md.traj')#                                                 azob_dis_no_consbiased_{args.shift}_large_new_cv_trigi_circular_mean


N_steps = int(1e5)
dyn.run(N_steps)
traj = read(f'azob_dis_no_consbiased_{args.shift}_large_new_cv_trig_rad.traj@2000::1')
cv_list=[]
energy_list=[]
with open(f"large_cv_energy_biased_dis_no_cons_{args.shift}_new_cv_trig_rad.csv", 'w') as file:
        writer =csv.writer(file)
        writer.writerow(["cv","energy"])
for atoms in traj:
    atoms.calc=biased_mace_calc
    energy=atoms.get_total_energy()
    cv= compute_cv(atoms.positions)
    with open(f"large_cv_energy_biased_dis_no_cons_{args.shift}_new_cv_trig_rad.csv", 'a+') as file:
        writer =csv.writer(file)
        writer.writerow([cv,energy])
    cv_list.append(cv)
    energy_list.append(energy)
    
with open(f"large_mean_cv_energy_dis_no_cons_{args.shift}_new_cv_trig_rad.csv", 'w') as file:
    writer =csv.writer(file)
    writer.writerow([x0,circular_mean(np.array(cv_list)),-k_*(circular_mean(np.array(cv_list))-x0),np.mean(np.array(energy_list)),compute_cv(atoms1.positions)])
        
    
