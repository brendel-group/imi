#!/bin/bash

# Script to collect FVs for all units without diversity.

# Call this as ./5_start_nodiv_opt_stim_collection.sh output_221031 units_resnet50_100_42_221031 resnet50/resnet50-l2/googlenet

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

# stitch them together into filename
filename=${output}/${unitsfile}.json

if [ ! -f "${filename}" ]; then
  echo "Unitsfile ${filename} does not exist! Aborting..."
  exit 1
fi

create_stimuli () {
  echo "Creating nodiv-stimuli for model ${1} and unit-file ${2}."
  sbatch run_gen_opt_stim_no_div.sh ${1} ${2} ${output}/${1}/large_alpha ${output}/${1}/skipped
}

# create stimuli for l2 resnet
create_stimuli ${model} ${filename}
