#!/bin/bash

# Script to sample a few units, collect activations, extract exemplars and obtain FVs.
# To sample many units, use the scripts in /parallel.

# argument 1: str, model-name
# argument 2: int, number of units
# argument 3: str, path to file in which units are stored

cd ~/interpretability-comparison/stimuli_generation

# skipping this step, because we already have a units-json-file
# generate json file that lists all units of interest for raw trials
# python3 sample_units.py --model_name ${1} --n_units ${2} --filename ${3}

# first, record activations for all units
python3 collect_activations.py --model ${1}

# second, extract exemplars and query images from 90th percentile
python3 extract_exemplars.py --model ${1} --units_file ${3} --start_min 5000 --stop_max 45000 --num_batches 20

# generate optimized stimuli
python3 get_diverse_optimized_stimuli.py --model_name ${1} --units_file ${3} --large_alpha_file large_alpha --skipped_units_file skipped_units
