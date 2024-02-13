## Parallel Stimuli Generation

This folder contains scripts and utility python files to collect stimuli for the full experiment in parallel.

Usage: Run `start_parallel_stimuli_generation.sh outputs` where the argument is the name of the folder in which to store status-files.

When sampling units, we use the format `units_{networkname}_{number of units}_{seed}_{date}.json` to create unique file names.

Some units might be skipped, because the FV-creation runs into problems. 
For other units, the largest attempted alpha value might be too small.
These units are collected in JSON files in the output folder and can be joined using `join_jsons.py`, to then start a new run over these units with larger values for alpha (skipped units should be dropped, happens only very rarely).