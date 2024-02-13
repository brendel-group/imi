#!/bin/bash

# arguments
# 1: model_name
# 2: full name to json file
# 3: filename for units with too large alpha
# 4: filename for units that had to be skipped

#SBATCH --job-name=no_div         # Job name
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=0-06:00            # Runtime in D-HH:MM
#SBATCH --partition=gpu-2080ti    # Partition to submit to
#SBATCH --mem=40G                 # Memory pool for all cores
#SBATCH --gres=gpu:1              # Request one GPU
#SBATCH --constraint=ImageNet2012 # Request a node that has ImageNet locally
#SBATCH --cpus-per-task=8         # Request all 8 CPUs

scontrol show job=$SLURM_JOB_ID

srun \
singularity exec \
--nv \
--bind /mnt/qb/work/bethge/tklein16/torch_models:/torch_models/ \
--bind /mnt/qb/work/bethge/tklein16/int_comp_out:/output/ \
--bind /scratch_local/datasets/ImageNet2012/:/imagenet/ \
/mnt/qb/work/bethge/tklein16/containers/int_comp.sif \
./gen_opt_stim_no_div.sh ${1} ${2} ${3} ${4}

echo DONE.
