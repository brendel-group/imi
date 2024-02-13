#!/bin/bash

# Script to collect FVs for multiple units in parallel (expected to be run on headnode) across num_gpus GPUs.

# Call this as ./4_start_opt_stim_collection.sh output_221031 units_resnet50_100_42_221031 resnet50/resnet50-l2/googlenet

# read in output directory
output=${1}

# read in units file (name without .json extension)
unitsfile=${2}

# read in model name
model=${3}

if [ -z "${output}" ] || [ -z "${unitsfile}" ] || [ -z "${model}" ]; then
  echo "At least one of the inputs was empty! Model, output dir and unitsfile need to be specified!"
  exit 1
fi

filename=${output}/${unitsfile}
num_gpus=25

if [ ! -f "${filename}.json" ]; then
  echo "Unitsfile ${filename}.json does not exist! Aborting..."
  exit 1
fi

if [ ! -f "${filename}_0.json" ]; then
  echo "Looks like units were not split! Aborting..."
  exit 1
fi

create_stimuli () {
  echo "Creating stimuli for model ${1} and unit-file ${2}.json across ${3} GPUs."

  # spread the collection of stimuli out across GPUs
  for ((gpu=0;gpu<${3};gpu++)); do
      echo "Dispatching FV-creation for model ${1} on GPU ${gpu}."
      sbatch run_gen_opt_stim.sh ${1} ${2}_${gpu}.json ${output}/${1}/skipped_${gpu} ${gpu}
  done
}

# create stimuli for l2 resnet
create_stimuli ${model} ${filename} ${num_gpus}
