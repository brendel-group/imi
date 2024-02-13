#!/bin/bash

# Script to collect activations.

# 1: model_name
# 2: num_gpus
# 3: gpu

python3 ../collect_activations.py --model ${1} --n_chunks ${2} --layer_chunk ${3}

# For ConvNext, we have too many layers, so only collect activations for layers which we know we'll sample from:

# python3 ../collect_activations.py --model ${1} --n_chunks ${2} --layer_chunk ${3} --units_file /home/bethge/tklein16/interpretability-comparison/stimuli_generation/parallel/output_230424/units_convnext_b_100_42_230424.json