import pandas as pd 
import numpy as np

df = pd.read_csv('combined_large_mean_cv_energy_dis_trig.csv')

df["prob"]= 2*(df["mean"]-df["pos"])

#write the dataframe to a csv file
df.to_csv('combined_large_mean_cv_energy_dis_trig_edited.csv', index=False)