#!/bin/bash

# Demo script to generate the experiments.
# This assumes stimuli for at least 84 units for the models in stimuli/.

# location where stimuli should be found
root_location=/path/to/data
stimuli_location=${root_location}/stimuli

# location where outputs should be written to
output_location=${root_location}/experiment
structure_location=${output_location}/structures
trial_location=${output_location}/exp_data

mkdir -p ${trial_location}
mkdir -p ${structure_location}

declare -a models=(
    googlenet
    resnet50
    resnet50_l2
    clip-resnet50
    wide_resnet50
    densenet_201
    convnext_b
    clip-vit_b32
    in1k-vit_b32
)

declare -a conditions=(
    natural
    optimized
)

for model in ${models[@]}; do
    for condition in ${conditions[@]}; do
        # make json for model and condition -- 63 hits with 40 trials each across 84 units means each unit is seen 30 times
        python3 create_task_structure_json.py -s ${stimuli_location} -o ${structure_location}/${model}_${condition}.json -sc resources/catch_trials -nc 5 -nt 40 -nh 63 -c ${condition} --model ${model} -upm 84 -maxb 10

        # generate the actual trials
        python3 create_task_structure_from_json.py -t ${trial_location}/${model}_${condition} -i ${structure_location}/${model}_${condition}.json -nr 9

        # copy instructions over
        cp -r resources/instructions/${condition}_9_references ${trial_location}/${model}_${condition}/instructions

        # copy json file into experiment folder
        cp ${structure_location}/${model}_${condition}.json ${trial_location}/${model}_${condition}/${model}_${condition}.json
    done
done