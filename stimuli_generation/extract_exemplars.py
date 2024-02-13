"""
Like extract_exemplars.py, but this extracts reference images from the top and query
images from the given percentile range.
"""

import os
import argparse

import numpy as np
import pandas as pd

from stimuli_generation.utils import (
    read_pickled_activations_file,
    read_units_file,
    split_unit,
    transform_and_copy_img,
    STORAGE_DIR,
    IMAGENET_PATH,
    accuracies,
)


def make_fair_batches(paths, num_lists: int, reverse=False):
    """
    Makes batches of natural stimuli from a list of paths, sorted ascending by
    activation (i.e. so that the most activating image is the last in paths).

    :param paths: list of paths, sorted ascending by absolute value of activations
    :param num_lists: how many fair lists to generate
    :param reverse: whether to reverse paths first, this is done for minima
    :returns: num_lists lists of batches
    """

    if reverse:
        paths.reverse()

    elems_per_list = int(len(paths) / num_lists)  # usually 10, i.e. 9 ref + 1 query

    # Create elems_per_list bins
    # (we need num_lists elements per bin, because each list will get one element
    # from each bin).
    bins = [paths[i * num_lists: (i + 1) * num_lists] for i in range(elems_per_list)]

    # shuffle every bin
    for data_bin in bins:
        np.random.shuffle(data_bin)

    # construct fair lists by taking the i-th value from every bin
    # note that the last image, which will be {min/max}_9, is taken from the best bin
    fair_vals = [[bins[j][i] for j in range(elems_per_list)] for i in range(num_lists)]

    return fair_vals


def extract_stimuli_range(dataf, unit, start_min, stop_min, start_max, stop_max):
    """
    Extracts stimuli for the given unit and the given range, returns them as sorted
    list from least to most activating.

    :param df: the pandas dataframe of activation values
    :param unit: the index of the unit
    :param start_min: start index for minima
    :param stop_min: end index for minima
    :param start_max: start index for maxima
    :param stop_max: stop index for maxima
    """

    # select only this unit and sort in ascending order
    unit_df = dataf[["path", str(unit)]]
    unit_df = unit_df.sort_values(str(unit), ascending=True)

    assert start_min < stop_min < start_max < stop_max, "Indices not reasonable!"
    assert (
        len(unit_df) >= stop_max
    ), f"Not enough activations for unit {unit} and index {stop_max}!"

    # Select the first few exemplars = minima, then make fair lists but reverse them
    # first so that min_9 is the strongest negatively activating image
    min_exemplars = unit_df.iloc[start_min:stop_min]["path"].tolist()
    max_exemplars = unit_df.iloc[start_max:stop_max]["path"].tolist()

    return min_exemplars, max_exemplars


def read_activations_file(args, activations_dir, layer, units):
    """
    Reads the file (.csv or .pkl) of activations and returns the relevant units as pandas DataFrame.

    :param args: the CLI arguments
    :param activations_dir: path to directory where activations files lie
    :param layer: the layer we are recording
    :param units: list of ints, the units we care about

    :returns: a pandas DataFrame that for all units of a layer and all images of the dataset stores their activation
    """
    if args.csv:
        csv_path = os.path.join(activations_dir, layer + ".csv")
        assert os.path.exists(csv_path), f"Could not find path to csv: {csv_path}"
        # only loading columns for the chosen units and the filepath
        dataframe = pd.read_csv(csv_path, usecols=['path'] + [str(u) for u in units])
    else:
        pkl_path = os.path.join(activations_dir, layer + ".pkl")
        assert os.path.exists(
            pkl_path), f"Could not find path to pickled file: {pkl_path}"

        dataframe = read_pickled_activations_file(pkl_path, units)

    return dataframe


def extract_stimuli_for_layer(args, layer, units):
    """
    Extracts the stimuli for a given layer and its units.

    :param args: CLI arguments
    :param layer: the layer name
    :param units: list of units of this layer
    """

    stimuli_dir = os.path.join(STORAGE_DIR, "stimuli", f"{args.model}_{args.extra_name}")
    os.makedirs(stimuli_dir, exist_ok=True)

    activations_dir = os.path.join(STORAGE_DIR, "activations", args.model)
    assert os.path.exists(activations_dir), "Could not find directory with activations!"

    # get path to CSV of this layer and get df
    dataframe = read_activations_file(args, activations_dir, layer, units)

    num_images_total = len(dataframe)  # how many images there are in total
    print(f"Found {num_images_total} images in total for layer {layer}.")

    # make sure that there is no overlap between query and reference images
    assert (
        arguments.num_batches * 9 <= arguments.start_min
    ), "Illegal combination of arguments! Queries and References would overlap!"
    assert (
        num_images_total - (arguments.num_batches * 9) > arguments.stop_max
    ), "Illegal combination of arguments! Queries and References would overlap!"

    # for each unit, find maximally/minimally activating images from chosen percentile
    for unit in units:

        # extract query images from given range
        min_queries, max_queries = extract_stimuli_range(
            dataframe,
            unit,
            args.start_min,
            args.stop_min,
            args.start_max,
            args.stop_max,
        )

        # extract 99 reference images from top and bottom (for 11 batches with 9 refs)
        min_refs, max_refs = extract_stimuli_range(
            dataframe,
            unit,
            0,
            args.num_batches * 9,
            num_images_total - (args.num_batches * 9),
            num_images_total,
        )

        # Combine the lists - both lists go from least to most, so min list starts with
        # queries (the first / last ten images land in the batch from which queries are
        # sourced, so this is fine).
        min_exemplars = min_queries + min_refs
        max_exemplars = max_refs + max_queries

        min_lists = make_fair_batches(min_exemplars, args.num_batches, reverse=True)
        max_lists = make_fair_batches(max_exemplars, args.num_batches)

        # for each unit, we create ten folders...
        for batch, (min_list, max_list) in enumerate(zip(min_lists, max_lists)):

            save_dir = os.path.join(
                stimuli_dir,
                layer,
                f"channel_{unit}",
                "natural_images",
                f"batch_{batch}",
            )
            os.makedirs(save_dir, exist_ok=True)

            # Images sorted from least to most activating would have names min_8
            # to max_8 with ten images each (idx 0 to 9)
            for i, (min_source_path, max_source_path) in enumerate(
                zip(min_list, max_list)
            ):

                min_path = os.path.join(save_dir, f"min_{i}.png")
                max_path = os.path.join(save_dir, f"max_{i}.png")

                transform_and_copy_img(
                    os.path.join(IMAGENET_PATH, min_source_path), min_path
                )
                transform_and_copy_img(
                    os.path.join(IMAGENET_PATH, max_source_path), max_path
                )


def main(args):
    """
    Extracts all images for the specified combination of model, layer and neurons.

    :param args: the CLI-arguments
    """

    # set random seed
    np.random.seed(arguments.seed)

    # get all the layers we need and map them to their respective units
    layer_map = {}
    for unit in args.units:
        layer, unit_idx = split_unit(unit)
        if layer in layer_map:
            layer_map[layer].append(unit_idx)
        else:
            layer_map[layer] = [unit_idx]

    for layer, units in layer_map.items():
        extract_stimuli_for_layer(args, layer, units)


if __name__ == "__main__":

    # Images are sorted in ascending order for each neuron, so 0-110 are the 100
    # least activating images
    parser = argparse.ArgumentParser(
        description="Extracting the minimally / maximally activating images for a list "
        "of units, from start to stop."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(accuracies.keys()),
        help="Which model to use.",
    )
    parser.add_argument(
        "--units_file",
        type=str,
        required=True,
        help="The json file of units to select.",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=11,
        help="The number of batches to collect for each unit.",
    )

    # start / stopping points are indices in a list sorted in ascending order
    parser.add_argument(
        "--start_min",
        type=int,
        help="Starting point for minima selection",
        required=True,
    )
    parser.add_argument(
        "--stop_max",
        type=int,
        help="Stopping point for maxima selection",
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random Seed for pseudorandom numbers used in shuffling of buckets",
        default=42,
    )
    parser.add_argument(
        '--csv',
        action="store_true",
        help="Whether the old method of reading from CSV files should be used.",
    )
    parser.add_argument(
        "--extra_name",
        type=str,
        default="",
        help="Extra name to be added to the model-name in the storage location for stimuli."
    )
    arguments = parser.parse_args()

    arguments.units = read_units_file(arguments.units_file)

    # calculate stopping points from number of batches
    arguments.stop_min = arguments.start_min + arguments.num_batches
    arguments.start_max = arguments.stop_max - arguments.num_batches

    # extract images
    main(arguments)
