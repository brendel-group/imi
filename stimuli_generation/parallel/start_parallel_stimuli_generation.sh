#!/bin/bash

# Script to collect multiple units in parallel (expected to be run on interactive partition)
# First, we collect 100 units (for resnet50 and resnet50-l2, we can use the same units because architecture is the same.)
# We then split the list of units into num_gpus fair lists (fair in the sense that they should take equally long.)
# Then, we collect FVs for all units across num_gpus GPUs and collect exemplars.

# read in output directory
output=${1}

# read in units file (name without .json extension)
unitsfile=${2}

if [ -z "${output}" ] || [ -z "${unitsfile}" ]; then
  echo "At least one of the inputs was empty! Output dir and unitsfile need to be specified! Aborting..."
  exit 1
fi

# check if output directory exists
if [ -d "${output}" ]; then
  echo "${output} exists already, aborting..."
  exit 1
fi

# create directory for status outputs (results are stored in $WORK)
mkdir ${output}
mkdir ${output}/resnet50
mkdir ${output}/resnet50-l2

model=resnet50
num_units=100
filename=${output}/${unitsfile}
num_gpus=25

# sample units
python3 ../sample_units.py --model_name ${model} --n_units ${num_units} --filename ${filename}.json

# split units into sub-lists
python3 ../split_units_list.py --model_name ${model} --units_file ${filename}.json --num_gpus ${num_gpus}

create_stimuli () {

  echo "Creating stimuli for model ${1} and unit-file ${2}.json across ${3} GPUs."

  # spread the collection of stimuli out across GPUs
  for ((gpu=0;gpu<${3};gpu++)); do
      echo "Dispatching FV-creation for model ${1} on GPU ${gpu}."
      sbatch run_gen_opt_stim.sh ${1} ${2}_${gpu}.json ${output}/${1}/large_alpha_${gpu} ${output}/${1}/skipped_${gpu} ${gpu}

  done

  # record activations and extract exemplars for all units
  sbatch run_collect_exemplars.sh ${1} ${2}.json

}

# create stimuli for standard resnet
create_stimuli ${model} ${filename} ${num_gpus}

# create stimuli for l2 resnet
model=resnet50-l2
create_stimuli ${model} ${filename} ${num_gpus}
