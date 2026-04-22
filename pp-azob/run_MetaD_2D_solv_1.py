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


import jax.numpy as jnp

def compute_dihedral_and_angle(positions):
    """
    Calculate the dihedral angle defined by four atomic positions,
    and the bond angle defined by three atomic positions.
    
    Parameters:
        positions (jnp.ndarray): A (N, 3) array containing atomic positions.
        dihedral_idxs (tuple of int): indices of the four atoms for dihedral (default (1,6,7,8)).
        angle_idxs (tuple of int): indices of the three atoms for angle (default (1,6,7)).
        
    Returns:
        jnp.ndarray: Array([dihedral_deg, angle_deg])
    """
    # --- Unpack positions for dihedral ---
    p1, p2, p3, p4 = positions[jnp.array([2,10,11,12])]
    
    # bond vectors for dihedral
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    
    # normal vectors
    n1 = jnp.cross(b1, b2)
    n2 = jnp.cross(b2, b3)
    
    # normalize
    n1 = n1 / jnp.linalg.norm(n1)
    n2 = n2 / jnp.linalg.norm(n2)
    b2_unit = b2 / jnp.linalg.norm(b2)
    
    # dihedral computation
    cos_phi = jnp.dot(n1, n2)
    sin_phi = jnp.dot(jnp.cross(n1, n2), b2_unit)
    phi = jnp.arctan2(sin_phi, cos_phi)
    dihedral_deg = jnp.degrees(phi)
    
    # --- Unpack positions for bond angle ---
    pa, pb, pc = positions[jnp.array([10, 11, 12])]
    
    # vectors for angle at atom b
    v1 = pa - pb
    v2 = pc - pb
    
    # normalize
    v1_u = v1 / jnp.linalg.norm(v1)
    v2_u = v2 / jnp.linalg.norm(v2)
    
    # angle computation
    cos_theta = jnp.dot(v1_u, v2_u)
    # clip to [-1,1] to avoid numerical issues
    cos_theta = jnp.clip(cos_theta, -1.0, 1.0)
    theta = jnp.arccos(cos_theta)
    angle_deg = jnp.degrees(theta)
    
    return jnp.array([dihedral_deg, angle_deg])


mace_calc_dis = mace_off(model="large",dispersion=False)
atoms1 = read("opt_solvated_pp-azob_trans_dmso_-25.xyz")#"azob_trans_opt.traj"
combinded_calc=mace_calc_dis
atoms = atoms1.copy()
atoms.calc = combinded_calc
BFGS(atoms).run(fmax=0.05,steps=500,)

#exit()ls me
timestep= 0.5*units.fs

MaxwellBoltzmannDistribution(atoms,temperature_K=300)
dyn = WT_Metadynamics(atoms, timestep=timestep, temperature_K=300, friction=.1, trajectory='metad_pp_azob_trans_2D_solv_cont_1_.traj',fixcm=False,
                   cvs=compute_dihedral_and_angle,std_dev=[5,5],bias_height=.1 ,interval_size=100,output_file='metad_pp_azob_trans_2D_solv_1.txt',input_file='metad_pp_azob_trans_2D_solv_1.txt',well_temp=True,bias_factor=10,wrapping=[True,True],bounds=((-180,180),(0,180)),max_bias=int(1e6))
dyn.run(7e5)
