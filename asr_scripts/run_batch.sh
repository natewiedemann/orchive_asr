#!/bin/bash

# Point this to the folder containing your sub-folders
PARENT_DIR="/net/DeepAL_data/data/tmp_orchive"

# Loop through each directory inside the parent directory
for d in "$PARENT_DIR"/*/; do
    dir_name=$(basename "$d")
    
    echo "Submitting job for directory: $dir_name"
    
    # Submit the slurm script, passing the full path of the sub-folder as $1
    sbatch batch_whisper.slurm "$d"
    sleep 1
done

echo "All jobs have been submitted to the queue."