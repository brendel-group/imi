#!/bin/bash

# Script to collect multiple units in parallel (expected to be run on interactive partition)
# First, we collect 100 units (for resnet50 and resnet50-l2, we can use the same units because architecture is the same.)
# We then split the list of units into num_gpus fair lists (fair in the sense that they should take equally long.)

# Calling this on interactive partition as ./1_start_unit_collection.sh output_221031 units_resnet50_100_42_221031 resnet50/resnet50-l2/googlenet

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

# check if output directory exists
if [ -d "${output}" ]; then
  echo "${output} exists already, aborting..."
  exit 1
fi

# create directory for status outputs (results are stored in $WORK)
mkdir ${output}
mkdir ${output}/${model}

num_units=100
filename=${output}/${unitsfile}
num_gpus=25

# sample units
python3 ../sample_units.py --model_name ${model} --n_units ${num_units} --filename ${filename}.json --seed 42

# split units into sub-lists
python3 split_units_list.py --model_name ${model} --units_file ${filename}.json --num_gpus ${num_gpus}
