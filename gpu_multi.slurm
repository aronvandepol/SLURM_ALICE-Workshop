#!/bin/bash
#SBATCH --job-name=gpu-test
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user=a.m.van.de.pol@hum.leidenuniv.nl
#SBATCH --mail-type="ALL"
#SBATCH --mem=5G
#SBATCH --time=00:5:00
#SBATCH --partition=gpu-short
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus=4


# load modules (assuming you start from the default environment)
# we explicitly call the modules to improve reproducibility
# in case the default settings change
module load Python

# Source the Python virtual environment
source $HOME/torch/bin/activate

echo "[$SHELL] #### Starting Python test"
echo "[$SHELL] ## This is $SLURM_JOB_USER on $HOSTNAME and this job has the ID $SLURM_JOB_ID"
# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] ## current working directory: "$CWD

# Run the file
echo "[$SHELL] ## Run script"
python3 gpu_test.py
echo "[$SHELL] ## Script finished"

echo "[$SHELL] #### Finished Python test. Have a nice day"