"""
Utility functions for data generation.
"""

import os
import random
import numpy as np


def get_and_update_batch_id(batch_id_dict, unit, min_batch_id, max_batch_id):
    """
    Utility function for obtaining the next batch ID that should be used for a unit.

    :param batch_id_dict: dictionary mapping units to their next batch-id
    :param unit: the unit we are interested in
    :param min_batch_id: the minimum batch index
    :param max_batch_id: the maximum batch index
    """
    next_id = batch_id_dict[unit]

    candidate = (next_id + 1) % (max_batch_id + 1)  # increase by one for next time
    candidate = max(min_batch_id, candidate)  # start at min_batch_id again
    batch_id_dict[unit] = candidate

    return next_id


def create_natural_trials(units, get_batch_id):
    """
    Creates list of trials with natural stimuli.

    :param units: list of str, the units to be used
    :param get_batch_id: callable that takes a unit and returns the batch id that
        should be used for this unit
    """

    trials = []
    for unit in units:
        batch_id = get_batch_id(unit)
        trials.append(
            dict(
                queries=os.path.join(unit, f"batch_{batch_id}"),
                references=os.path.join(unit, f"batch_{batch_id}"),
                mode="natural",
            )
        )

    random.shuffle(trials)
    return trials


def create_optimized_trials(units, get_batch_id):
    """
    Creates list of trials with optimized stimuli.

    :param units: list of str, the units to be used
    :param get_batch_id: callable that takes a unit and returns the batch id that
        should be used for this unit
    """

    def get_dirname(u, query):
        """
        For optimized condition, but hard-mode, we need to change file paths so that they point to the folder
        of the main model.
        TODO after neurips deadline, I should probably change this logic, this is a quick-and-dirty fix.

        u: unit 
        query: True if query image
        """
        dirname = os.path.dirname(u)
        if query:
            # i.e. natural image
            return dirname
        else:
            # optimized image, which is in other location
            # e.g. .../stimuli/model/layer/channel_xx/
            elems = dirname.split("/")
            channel = elems[-1]
            layer = elems[-2]
            model = elems[-3]
            if "_hard" in model:
                new_model = model.split("_hard")[0]
                dirname = "/" + os.path.join(os.path.join(*elems[:-3]), new_model, layer, channel)
                print("dirname for references:", dirname)
            return dirname

    trials = [
        dict(
            queries=os.path.join(
                get_dirname(unit, True), "natural_images", f"batch_{get_batch_id(unit)}"
            ),
            references=os.path.join(get_dirname(unit, False), "optimized_images"),
            mode="optimized",
        )
        for unit in units
    ]

    random.shuffle(trials)
    return trials


def create_catch_trials(available_units, num_catch_trials):
    """
    Creates trials for a task from the set of available units for catch trials.

    :param available_units: list of str, units available for catch trials
    :param num_catch_trials: int, how many catch trials to generate
    """

    # If we have fewer units than we want to create catch trials, use all once, then
    # sample the rest with replacement.
    if len(available_units) < num_catch_trials:
        chosen_units = available_units.copy()
        chosen_units.extend(
            np.random.choice(
                available_units,
                size=num_catch_trials - len(available_units),
                replace=True,
            )
        )
    else:
        chosen_units = np.random.choice(
            available_units, size=num_catch_trials, replace=False
        )

    return [
        dict(
            queries=os.path.join(os.path.dirname(unit), "natural_images"),
            references=unit,
            mode="catch_trial",
        )
        for unit in chosen_units
    ]
