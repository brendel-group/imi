"""
This script creates a json-file with the structure of the tasks for the experiment.

The json file contains a dictionary (called data) with one key ('tasks') and value
(a list).

This list contains n_tasks dictionaries with four entries:
'index'                 indicating the index of the task
'raw_trials'            contains a list with the data of --n-trials trials
'raw_catch_trials'      contains a list with the data of --n-catch-trials catch trials
'trials'                contains a list with the shuffled and merged data of raw_trials
and raw_catch_trials

Each list item contains a dictionary with the entries 'mode'
('catch_trial' or 'natural' / 'optimized'), 'queries' (absolute paths)
and 'references' (absolute paths), e.g.:
{'mode': 'optimized',
'queries': '/path/to/stimuli/model_name/{layer_name}/channel_{number}/
natural_images/batch_2',
'references': '/path/to/stimuli/model_name/{layer_name}/channel_{number}/
optimized_images'}

NOTE: You can choose how many units should be made available for each model, and how
many trials to show per HIT.
These values might conflict with each other, and this script favours the trials,
i.e. it drops the units that can not be distributed evenly across trials.

NOTE: An earlier version of this script was used to construct experiments for several models,
but it contained a bug, causing the selection of only 80 of the 84 available units.
This bug is fixed here (see git history for the old version).
The old version of the script was then also used to create the experiments for the hard condition,
so that the units between these conditions match.
"""

import argparse
import glob
import os
import random
import json
import numpy as np

from tools.data_generation.datagen_utils import (
    create_catch_trials,
    create_natural_trials,
    create_optimized_trials,
    get_and_update_batch_id,
)


def get_units_for_trials(source_folder, model, extra_name, condition, upm):
    """
    Obtains all units available for trials by stitching together source folder and path,
    depending on mode.

    :param source_folder: path to stimuli
    :param model: str, model name
    :param extra_name: str, what should be added to the model-name
    :param condition: str, 'natural' or 'optimized'
    :param upm: int, units per model
    """

    all_units = glob.glob(
        os.path.join(
            source_folder,
            f"{model}_{extra_name}",
            "*",
            "channel_*",
            f"{condition}_images"
        )
    )
    # Shuffle, just to make sure that order of layers is random.
    random.shuffle(all_units)
    all_units = all_units[:upm]  # only keep as many as desired by user

    print(f"We have {len(all_units)} units available for trials.")
    return all_units


def get_units_for_catch_trials(source_folder, condition):
    """
    Obtains all units available for catch trials.

    :param source_folder: path to folder with catch-trial-stimuli
    """
    units = glob.glob(
        os.path.join(source_folder, "*", "channel*", f"{condition}_images")
    )
    print(f"We have {len(units)} units available for catch trials.")
    return units


def main(args):
    """
    Entry point for the generation of json-structures.

    :param args: the CLI-args
    """

    # assert that the number of units we want to use per model can be distributed fairly
    assert (args.n_hits * args.n_trials) % args.units_per_model == 0
    n_unit_repetitions = int(args.n_hits * args.n_trials / args.units_per_model)
    print(f"Every unit will be shown {n_unit_repetitions} times.")

    # find all units for which we have stimuli
    units_available_for_trials = get_units_for_trials(
        args.source_folder, args.model, args.extra_name, args.condition, args.units_per_model
    )
    random.shuffle(units_available_for_trials)

    units_available_for_catch_trials = get_units_for_catch_trials(
        args.source_catch_folder, args.condition
    )
    random.shuffle(units_available_for_catch_trials)

    # We might have more units available than we need, so we prune the lists
    units_available_for_catch_trials = units_available_for_catch_trials[
        : args.n_catch_trials
    ]

    # select function based on condition
    trial_func = (
        create_natural_trials
        if args.condition == "natural"
        else create_optimized_trials
    )

    # data structure and function to use batch-ids evenly - this is over-engineered,
    # but maybe we need the flexibility
    batch_id_dict = {unit: args.min_batch_id for unit in units_available_for_trials}

    def get_batch_id(unit):
        return get_and_update_batch_id(
            batch_id_dict, unit, args.min_batch_id, args.max_batch_id
        )

    # the dict that will hold task-structures
    data = dict(tasks=[])

    # create backlog of units that need to be used
    backlog = units_available_for_trials.copy()

    # make tasks
    for task_idx in range(args.n_hits):

        if len(backlog) >= args.n_trials:
            units_chosen_for_trials = backlog[:args.n_trials]
            backlog = backlog[args.n_trials:]
        else:
            units_chosen_for_trials = backlog.copy()
            backlog = units_available_for_trials.copy()
            random.shuffle(backlog) 

            while len(units_chosen_for_trials) < args.n_trials:
                candidate = random.choice(backlog)
                if candidate not in units_chosen_for_trials:
                    units_chosen_for_trials.append(candidate)
                    backlog.remove(candidate)

        if not backlog:
            backlog = units_available_for_trials.copy()
            random.shuffle(backlog)    

        assert len(units_chosen_for_trials) == args.n_trials, "Wrong number of units selected!"
        assert len(set(units_chosen_for_trials)) == args.n_trials, "Unit was selected twice!"

        # shuffle them, so that there is no systematic relationship in their order
        random.shuffle(units_chosen_for_trials)

        # each task is represented as a dict that maps 'raw_trials' to a dictionary
        # describing the task and its trials
        task = dict(index=task_idx + 1)

        # make raw trials with natural stimuli, but use the same batch for all tasks
        task["raw_trials"] = trial_func(units_chosen_for_trials, get_batch_id)

        # make catch trials = trials to catch people who are not paying attention
        task["raw_catch_trials"] = create_catch_trials(
            units_available_for_catch_trials,
            args.n_catch_trials,
        )

        # join these trials and shuffle them randomly
        task["trials"] = task["raw_trials"] + task["raw_catch_trials"]

        random.shuffle(task["trials"])

        data["tasks"].append(task)

    assert len(backlog) == len(units_available_for_trials), "Numbers didn't play out!"

    # dump task-structures to json-file
    with open(args.output, "w", encoding="ascii") as file:
        json.dump(data, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--source-folder",
        required=True,
        help="Path to source stimuli (/path/to/stimuli/).",
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Where to save final json structure."
    )
    parser.add_argument(
        "-sc",
        "--source-catch-folder",
        required="True",
        help="Path to stimuli to be used in catch-trials.",
    )
    parser.add_argument(
        "-nc",
        "--n-catch-trials",
        type=int,
        required=True,
        help="Number of catch trials per task.",
    )
    parser.add_argument(
        "-nt", "--n-trials", type=int, required=True, help="Number of trials per task."
    )
    parser.add_argument(
        "-nh", "--n-hits", type=int, required=True, help="Number of HITs (tasks)."
    )
    parser.add_argument(
        "-c",
        "--condition",
        required=True,
        type=str,
        choices=["natural", "optimized"],
        help="Which condition to use for trials.",
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        type=str,
        help="Which model to select from the stimuli folder.",
    )
    parser.add_argument(
        "--extra_name",
        type=str,
        default="",
        help="Extra name to be added to the model when finding stimuli."
    )
    parser.add_argument(
        "-upm",
        "--units-per-model",
        required=True,
        type=int,
        help="How many (of the available) units to choose.",
    )
    parser.add_argument(
        "-minb",
        "--min-batch-id",
        default=0,
        type=int,
        help="Minimum batch ID to use for each unit.",
    )
    parser.add_argument(
        "-maxb",
        "--max-batch-id",
        required=True,
        type=int,
        help="Maximum batch ID to use for each unit.",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="Seed for random number generator."
    )

    arguments = parser.parse_args()
    print("Received arguments:", arguments)

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)

    # generate task structures
    main(arguments)
