#!/bin/bash

# arguments
# 1: model_name
# 2: num_gpus
# 3: gpu

#SBATCH --job-name=collect_exem   # Job name
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=0-24:00            # Runtime in D-HH:MM
#SBATCH --partition=gpu-2080ti    # Partition to submit to, use gpu-v100 for densenet and convnext
#SBATCH --mem=40G                 # Memory pool for all cores
#SBATCH --gres=gpu:1              # Request one GPU
#SBATCH --constraint=ImageNet2012 # Request a node that has ImageNet locally
#SBATCH --cpus-per-task=8         # Request all 8 CPUs

# include information about the job in the output
scontrol show job=$SLURM_JOB_ID

# split the list of units into n-gpus many lists
srun \
singularity exec \
--nv \
--bind /mnt/qb/work/bethge/tklein16/torch_models:/torch_models/ \
--bind /mnt/qb/work/bethge/tklein16/int_comp_out:/output/ \
--bind /scratch_local/datasets/ImageNet2012/:/imagenet/ \
/mnt/qb/work/bethge/tklein16/containers/int_comp.sif \
./collect_activation.sh ${1} ${2} ${3}

echo DONE.
