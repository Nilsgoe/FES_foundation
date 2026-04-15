import pandas as pd
import os

# Initialize an empty list to store data from each file
data_frames = []

# Loop through the specified range and build file paths
for i in range(-4,35):
    file_name = f"large_mean_cv_energy_dis_{i}_new_cv.csv"
    
    # Check if the file exists
    if os.path.exists(file_name):
        # Read the file with a specified delimiter (comma by default)
        df = pd.read_csv(file_name, delimiter=',',header=None)

        
        # Confirm that the file has data in multiple rows
        print(f"{file_name} shape: {df.shape}")
        
        # Append the DataFrame to the list
        data_frames.append(df)
        
# Concatenate all data frames row-wise into one DataFrame
combined_df = pd.concat(data_frames, ignore_index=True)
print(f"{file_name} shape: {combined_df.shape}")
# Save the combined DataFrame to a new CSV file
combined_df.to_csv("combined_large_mean_cv_energy_dis_new_cv.csv", index=False)
