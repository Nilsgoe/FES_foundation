from Metadynamics import WT_Metadynamics

from biASE import General_Bias_Calculator

from mace.calculators import mace_off
import jax.numpy as jnp
from ase.io import read
from ase.optimize import BFGS
from ase.calculators.mixing import SumCalculator
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

def smooth_rising_function(x,height=25, center=90, boundary=105, steepness=20):
    # Adjusts the function to be approximately 0 within [center - boundary, center + boundary] and rise rapidly outside
    e_value = height*jnp.tanh(steepness * (jnp.abs(x - center) - boundary))
    return e_value[0]


def compute_cv(positions):
    """
    Calculate the dihedral angle defined by four atomic positions.
    
    Parameters:
        positions (jnp.ndarray): A (4, 3) array containing the positions of the four atoms.
    
    Returns:
        float: Dihedral angle in degrees.
    """
    # Extract positions of the atoms
    p1, p2, p3, p4 = positions[jnp.array([1,6,7,8])]# if trans :positions[jnp.array([2,11,12,13])] 
    #print(p1,p2,p3,p4)
    # Define vectors
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    
    # Calculate normal vectors of the planes
    n1 = jnp.cross(b1, b2)
    n2 = jnp.cross(b2, b3)
    
    # Normalize the normal vectors
    n1 /= jnp.linalg.norm(n1)
    n2 /= jnp.linalg.norm(n2)
    
    # Calculate unit vector along b2
    b2 /= jnp.linalg.norm(b2)
    
    # Calculate cosine and sine of the dihedral angle
    cos_phi = jnp.dot(n1, n2)
    sin_phi = jnp.dot(jnp.cross(n1, n2), b2)
    
    # Compute the angle in radians and convert to degrees
    phi = jnp.arctan2(sin_phi, cos_phi)
    return jnp.array([jnp.degrees(phi)])

mace_calc_dis = mace_off(model="large",dispersion=True,device="cuda:0")
atoms1 = read("azob_cis_opt.traj")#"azob_trans_opt.traj"
combinded_calc=mace_calc_dis
atoms = atoms1.copy()
atoms.calc = combinded_calc
BFGS(atoms).run(fmax=0.01,steps=500,)

#exit()ls me
timestep= 0.5*units.fs

MaxwellBoltzmannDistribution(atoms,temperature_K=333)
dyn = WT_Metadynamics(atoms, timestep=timestep, temperature_K=333, friction=.1, trajectory='metad_azob_cis_test.traj',fixcm=False,
                   cvs=compute_cv,std_dev=5,bias_height=.1,interval_size=100,output_file='metad_azob_cis_test.txt',well_temp=True,bias_factor=10)
dyn.run(1e6)
