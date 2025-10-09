#!/bin/bash
#
#SBATCH --job-name=DSprites_KTE_Array
#SBATCH --array=1-1200          # Run 1200 jobs, corresponding to the 1200 lines in your parameter file
#SBATCH --time=00:30:00         # Set a max run time (e.g., 30 minutes per job)
#SBATCH --mem=40G                # Request 4GB of memory per job
#SBATCH --cpus-per-task=1       # Request 1 CPU core per job
#SBATCH --output=slurm_logs/arr_%A_%a.out # Output file: %A=JobID, %a=ArrayTaskID

# ------------------------------------------------------------------
# 1. Setup Environment (if necessary, e.g., loading an Anaconda env)
# ------------------------------------------------------------------
echo "======================="
echo "Loading Anaconda Module..."
module load miniconda
source activate adaptive_exp

# ------------------------------------------------------------------
# 2. Extract Parameters for the current job index
#    The SLURM array index is 1-based, matching the CSV line number.
# ------------------------------------------------------------------
LINE_NUM=$SLURM_ARRAY_TASK_ID

# Use awk to select the parameters from the CSV file (skipping the header row)
# Note: $1, $2, $3 correspond to the columns in the CSV after the line number.
# We skip the 1st column (line number) and the 2nd (scenario) if we use the full array.
#
# The parameter file columns are: line,scenario,method,seed

# Get the full parameter line (skipping the header)
PARAMS=$(awk "NR==$LINE_NUM+1" experiment_parameters/dsprite_kte_binary_adaptive_parameters.csv)

# Use cut/awk with ',' as the delimiter to extract values
SCENARIO=$(echo $PARAMS | cut -d, -f2)
METHOD=$(echo $PARAMS | cut -d, -f3)
SEED=$(echo $PARAMS | cut -d, -f4)

# ------------------------------------------------------------------
# 3. Execute the Python script for this specific case
# ------------------------------------------------------------------
echo "Starting job $LINE_NUM: Scenario=$SCENARIO, Method=$METHOD, Seed=$SEED"

python experiments_dsprite.py \
    --run \
    --scenario "$SCENARIO" \
    --method "$METHOD" \
    --seed "$SEED"

echo "Job $LINE_NUM finished."