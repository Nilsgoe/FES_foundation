import numpy as np
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv("FES_UMD_azob.csv") #FES_UMD_malon.csv")
fes_umd = df['FES'].to_numpy()-1
cv_umd = df['CV'].to_numpy()/np.deg2rad(1)
o_file= "./viper/output_bias_1D_continued.txt"

scm = np.loadtxt(o_file, usecols=0)  # Collective variable values
fes = np.loadtxt(o_file, usecols=1) 
# Import free energy and reshape with the number of bins defined in the
# reconstruction process.
scm = np.loadtxt(o_file, usecols=0).reshape(501)#, 301)#was 201 everywhere
#tcm = np.loadtxt(o_file, usecols=1).reshape(301)#, 301)
fes = np.loadtxt(o_file, usecols=1).reshape(501)#, 301)

# Plot
fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(scm,fes,label="MetaD", linewidth=3.5)
ax.plot(cv_umd,fes_umd,label="UMD",linewidth=3.5)
# Plot parameters
ax.set_xlabel('CV',size=32)
ax.set_ylabel('FES',size=32)
ax.tick_params(axis='y', labelsize=24)
ax.tick_params(axis='x', labelsize=24)
ax.xaxis.set_major_locator(MaxNLocator(nbins=7, prune=None))
#cbar = fig.colorbar(im, ax=ax)
#cbar.set_label(label=r'FES[$\epsilon$]', fontsize=40)
#cbar.ax.tick_params(labelsize=38)
#plt.scatter()
ax.spines['top'].set_linewidth(3)
ax.spines['right'].set_linewidth(3)
ax.spines['bottom'].set_linewidth(3)
ax.spines['left'].set_linewidth(3)
plt.legend(fontsize=24)
plt.title("Dihedral Azob")
plt.tight_layout()
plt.savefig('FES_UMD_metad_azob.png',dpi=300)
plt.show()

