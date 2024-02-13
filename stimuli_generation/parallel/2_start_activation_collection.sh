#!/bin/bash

# Script to collect activations for all layers in parallel (expected to be run on login node)

# Call this as ./2_start_activation_collection.sh resnet50/resnet50-l2/googlenet

# read in model name
model=${1}

if [ -z "${model}" ]; then
  echo "No model was specified! Aborting..."
  exit 1
fi

num_gpus=3

collect_activations () {

  echo "Collecting activations for model ${1} across ${2} GPUs."

  # spread the collection of stimuli out across GPUs
  for ((gpu=0;gpu<${2};gpu++)); do
      echo "Dispatching activation collection for model ${1} on GPU ${gpu}."
      sbatch run_collect_activation.sh ${1} ${num_gpus} ${gpu}
  done
}

collect_activations ${model} ${num_gpus}
