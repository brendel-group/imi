#!/bin/bash

# argument 1: str, model-name
# argument 2: str, path to file in which units are stored
# argument 3: str, filename for units that had to be skipped
# argument 4: int, gpu ID

cd ~/interpretability-comparison/stimuli_generation

# generate optimized stimuli
python3 get_diverse_optimized_stimuli.py --model_name ${1} \
                                         --units_file parallel/${2} \
                                         --skipped_units_file parallel/${3} \
                                         --num_images 9 \
                                         --gpu ${4}
