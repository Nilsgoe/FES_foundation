#!/bin/bash

# Define the range you want to loop over (for example, 1 to 100)
for i in {-5..35}
do
  # Call the sbatch script and pass the number as an argument
  sbatch run_single_UMD.sh $i
done
