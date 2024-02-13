#!/bin/bash

# Script to extract exemplars.

# argument 1: str, model-name
# argument 2: str, path to file in which units are stored

# ImageNet training set has N=1,281,167 images

# sampling from the end of the distribution, 20 batches as in Borowski et al
# python3 ../extract_exemplars.py --model ${1} --units_file ${2} --start_min 180 --stop_max 1280986 --num_batches 20

# for medium experiment, sample from the 99% interval at index floor(N*0.01)=12811
# and high activating at index ceil(N*0.99)=1268356
#python3 ../extract_exemplars.py --model ${1} --units_file ${2} --start_min 12811 --stop_max 1268356 --num_batches 20 --extra_name hard99

# for hard experiment, samplfe from the 95% interval at index floor(N*0.05)=64058
# and high activating at index ceil(N*0.95)=1217109
#python3 ../extract_exemplars.py --model ${1} --units_file ${2} --start_min 64058 --stop_max 1217109 --num_batches 20 --extra_name hard95

# for very hard experiment, sample from the 85% interval at index floor(N*0.15)=192,175
# and high activating at index ceil(N*0.85)=1,088,992
# python3 ../extract_exemplars.py --model ${1} --units_file ${2} --start_min 192175 --stop_max 1088992 --num_batches 20 --extra_name hard85
