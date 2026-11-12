#!/bin/bash
#SBATCH --job-name=gmm_repeat       # Set a name for your job. This is especially useful if you have multiple jobs queued.
#SBATCH --partition=general         # Slurm partition to use
#SBATCH --qos=general  
#SBATCH --array=11-20               # How many tasks in the array
#SBATCH --time=11:50:00             # Wall time limit in D-HH:MM
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=out/gmm_repeat_%A_%a.out        # File to which STDOUT will be written
#SBATCH --error=out/gmm_repeat_%A_%a.err        # File to which STDERR will be written
#SBATCH --mail-type=ALL            # Type of email notification- NONE,BEGIN,END,FAIL,ALL
#SBATCH --mail-user=yikunz@uchicago.edu  # Email to which notifications will be sent

source ../yikun_venv/bin/activate

i=${SLURM_ARRAY_TASK_ID}
python3 ./GMM_Repeat_Sim.py ${i}
