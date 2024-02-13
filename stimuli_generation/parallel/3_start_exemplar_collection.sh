#!/bin/bash

# Script to collect multiple units in parallel (expected to be run on login node)
# Collect FVs for all units across num_gpus GPUs and collect exemplars.

# Calling this as ./3_start_exemplar_collection.sh output_221031 units_resnet50_100_42_221031 resnet50/resnet50-l2/googlenet

# read in output directory
output=${1}

# read in units file (name without .json extension)
unitsfile=${2}

# read in model name
model=${3}

if [ -z "${output}" ] || [ -z "${unitsfile}" ] || [ -z "${model}" ]; then
  echo "At least one of the inputs was empty! Model, output dir and unitsfile need to be specified! Aborting..."
  exit 1
fi

num_units=100
filename=${output}/${unitsfile}

if [ ! -f "${filename}.json" ]; then
  echo "Unitsfile ${filename}.json does not exist! Aborting..."
  exit 1
fi

create_stimuli () {
  echo "Extracting exemplars for model ${1} and unit-file ${2}.json."
  
  # record activations and extract exemplars for all units
  sbatch run_collect_exemplars.sh ${1} ${2}.json
}

create_stimuli ${model} ${filename}
