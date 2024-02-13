"""
Records the activations for a few chosen layers, for all neurons, for all validation
set images.
Writes these activations to one csv-file for each layer, so that we can extract
high / low - activating images later.

This script extracts all known layers of the chosen model, then splits it into
n_chunks chunks, then collects stimuli for one of the chunks.
"""

import os
import argparse
from datetime import datetime as dt
import pickle
import logging

import torch
import pandas as pd
import numpy as np
from lucent.optvis.render import ModelHook
from stimuli_generation.utils import (
    chunks,
    get_dataloader,
    get_relevant_layers,
    aggregate_activations,
    read_units_file,
    get_layers_from_units_list,
    get_model_layers,
    load_model,
    get_label_translator,
    accuracies,
    IMAGENET_PATH,
    STORAGE_DIR,
    DEVICE,
    get_clip_zero_shot_classifier,
    get_clip_logits
)


def record_activations_pickle(model, model_name, layer_names, target_path, use_validation_set):
    """
    Records the activation achieved by every image at every neuron, writes to npy-file.

    :param model: the pytorch model
    :param model_name: the name of the model as str
    :param layer_names: the names of the layers
    :param target_path: path to folder where CSV should be stored
    :param use_validation_set: whether to source exemplars from the validation set
    """
    logging.info(
        f"Recording activations for layers {layer_names}, will write to pkl-file.")

    results = {l: {'activations': None, 'paths': []} for l in layer_names}
    # `results` maps a layer name to a dictionary.
    # This dict maps "activations" to a n_imgs x n_units numpy array
    # and "paths" to a list of strings of length n_imgs

    def get_name(fname):
        # extracts the filename from a full filepath (to train image)
        return fname[len(IMAGENET_PATH):]

    source = "val" if use_validation_set else "train"
    dataloader = get_dataloader(os.path.join(IMAGENET_PATH, source), model_name)

    # for clip models, we need to construct a zero-shot classifier
    if "clip" in model_name:
        classifier = get_clip_zero_shot_classifier(model, model_name)

    if model_name == "googlenet":
        label_translator = get_label_translator()

    # store the number of correct and total samples for evaluating performance
    correct_samples = 0
    total_samples = 0

    with torch.no_grad() and ModelHook(model, layer_names=layer_names) as hook:

        start_idx = 0
        for batch_number, (batch, labels, paths) in enumerate(dataloader):

            batch = batch.to(DEVICE)
            labels = labels.to(DEVICE)

            if "clip" in model_name:
                logits = get_clip_logits(model, classifier, batch)
            else:
                logits = model(batch)

            if model_name == 'googlenet':
                assert logits.shape[-1] == 1008
                logits = logits[:, :1000]
                labels = label_translator(labels)

            # for getting the accuracy
            preds = torch.argmax(logits, -1)
            correct_samples += (preds == labels).sum()
            total_samples += labels.shape[0]

            for layer_name in layer_names:

                activations = hook(layer_name)

                activations = aggregate_activations(activations, model_name, layer_name)

                batchsize, units = activations.shape

                # to avoid copying around numpy arrays all the time, just allocate a huge one and fill it
                if results[layer_name]['activations'] is None:
                    num_rows = 50_000 if use_validation_set else 1_281_167
                    results[layer_name]['activations'] = np.zeros(
                        (num_rows, units), dtype=np.float32)

                # Update the activations and paths
                results[layer_name]['activations'][start_idx:start_idx +
                                                   batchsize, :] = activations.detach().cpu().numpy()
                results[layer_name]['paths'].extend([get_name(p) for p in paths])

            start_idx += batchsize

    # create activations.pkl-file for every layer
    for layer in layer_names:

        # construct descriptive file name
        filename = f"{layer}.pkl"
        full_path = os.path.join(target_path, filename)

        logging.info(f"Writing results to file for layer {layer}")
        with open(full_path, "wb") as fhandle:
            pickle.dump(results[layer], fhandle)

    # making sure that validation set accuracy had the right value
    acc = correct_samples / total_samples
    logging.info(
        f"{model_name} achieved {acc * 100}% {'validation' if use_validation_set else 'training'} set accuracy.")

    if use_validation_set:
        eps = 0.01
        assert accuracies[model_name] - eps <= acc <= accuracies[model_name] + eps, \
            f"Accuracy ({acc}) was not within tolerance of {accuracies[model_name]}"


def record_activations(model, model_name, layer_names, target_path, use_validation_set):
    """
    Records the n strongest activating images for a neuron.

    :param model: the pytorch model
    :param model_name: the name of the model as str
    :param layer_names: the names of the layers
    :param target_path: path to folder where CSV should be stored
    :param use_validation_set: whether to source exemplars from the validation set
    """
    logging.info(f"Recording activations for layers {layer_names}")

    results = {
        ln: [] for ln in layer_names
    }  # storing data before adding to df, because adding to df is expensive
    # See https://stackoverflow.com/questions/10715965/
    # create-a-pandas-dataframe-by-appending-one-row-at-a-time

    def dump_results():
        """Dumps the current set of results to file, to free up RAM."""

        # create activations.csv-file for every layer
        for layer in layer_names:

            # construct descriptive file name
            filename = f"{layer}.csv"
            full_path = os.path.join(target_path, filename)

            dataframe = pd.DataFrame(results[layer])

            # write to csv, append if necessary
            if os.path.exists(full_path):
                dataframe.to_csv(full_path, mode="a", header=False, index=False)
            else:
                dataframe.to_csv(full_path, mode="w", header=True, index=False)

    def get_name(fname):
        # extracts the filename from a full filepath (to train image)
        return fname[len(IMAGENET_PATH):]

    source = "val" if use_validation_set else "train"
    dataloader = get_dataloader(os.path.join(IMAGENET_PATH, source), model_name)

    # store the number of correct and total samples for evaluating performance
    correct_samples = 0
    total_samples = 0

    # for clip models, we need to construct a zero-shot classifier
    if "clip" in model_name:
        classifier = get_clip_zero_shot_classifier(model, model_name)

    if model_name == "googlenet":
        label_translator = get_label_translator()

    with torch.no_grad() and ModelHook(model, layer_names=layer_names) as hook:

        for batch_number, (batch, labels, paths) in enumerate(dataloader):

            batch = batch.to(DEVICE)
            labels = labels.to(DEVICE)

            if "clip" in model_name:
                logits = get_clip_logits(model, classifier, batch)
            else:
                logits = model(batch)

            # Lucent / Lucid InceptionV1 outputs 1008 classes with permuted labels
            if model_name == 'googlenet':
                assert logits.shape[-1] == 1008
                logits = logits[:, :1000]
                labels = label_translator(labels)

            # for getting the accuracy
            preds = torch.argmax(logits, dim=-1)
            correct_samples += (preds == labels).sum()
            total_samples += labels.shape[0]

            for layer_name in layer_names:

                activations = hook(layer_name)

                activations = aggregate_activations(activations, model_name, layer_name)

                _, units = activations.shape

                # walk over all images of the batch
                for i, path in enumerate(paths):
                    res_dict = {
                        neuron: act.item()
                        for neuron, act in zip(range(units), activations[i, :])
                    }
                    res_dict["path"] = get_name(path)
                    results[layer_name].append(res_dict)

            # store results every 100 iterations, to not run out of RAM
            if batch_number != 0 and batch_number % 100 == 0:
                logging.info(
                    f"Dumping results in iteration {batch_number} at {str(dt.now())}"
                )
                dump_results()
                del results
                results = {ln: [] for ln in layer_names}

        # dump the last results to file
        dump_results()

    # making sure that validation set accuracy has the right value
    acc = correct_samples / total_samples
    logging.info(
        f"{model_name} achieved {acc * 100}% {'validation' if use_validation_set else 'training'} set accuracy."
    )

    if use_validation_set:
        eps = 0.01
        assert (
            accuracies[model_name] - eps <= acc <= accuracies[model_name] + eps
        ), f"Accuracy ({acc}) was not within tolerance of {accuracies[model_name]}"


def main(model, model_name, layer_names, use_validation_set, csv):
    """
    Loads model, then feeds all ImageNet validation set images through the model,
    making sure that
    the validation set performance matches the reported performance. Then, writes a
    large .csv-file
    for every chosen layer of the network, in which the activations for all images at
    this layer are recorded.

    :param model_name: the model name
    :param layer_names: the names of the layers
    :param use_validation_set: whether to collect stimuli from the validation set
    :param csv: whether the old method of writing to CSV should be used
    """

    # create target directory
    target_path = os.path.join(STORAGE_DIR, 'activations', model_name)
    os.makedirs(target_path, exist_ok=True)

    # check if the model actually has this layer, and no activation-file already exists
    all_layers = get_model_layers(model)
    remaining_layer_names = []
    for layer_name in layer_names:

        assert layer_name in all_layers, f"Model {model_name} has no layer {layer_name}"

        # if the file with activations already exists, skip this layer
        suffix = ".csv" if csv else ".pkl"
        if not os.path.exists(os.path.join(target_path, layer_name + suffix)):
            remaining_layer_names.append(layer_name)

    logging.info(f"Found {len(remaining_layer_names)} remaining layers: {remaining_layer_names}")

    if csv:
        # Record all activations for the model and validation set, for all layers of interest,
        # and store to CSV (only works for small models).
        record_activations(model, model_name, remaining_layer_names,
                           target_path, use_validation_set)
    else:
        # Doing it layer by layer with multiple forward passes, because of 40G RAM limit.
        # Densenet is too big for 2080ti, so using V100 with enough VRAM to do 4 layers at once.
        for layer_subset in chunks(remaining_layer_names, 4 if model_name in ['densenet_201', 'convnext_b'] else 2):
            record_activations_pickle(
                model, model_name, layer_subset, target_path, use_validation_set)


def get_subset(layers, chunks, idx):
    """
    Splits a list of layers into chunks and returns one of them.

    :param layers: list of str, layers of the model
    :param chunks: how many chunks to create
    :param idx: int, index of the chunk to use here
    """

    # how many layers are in one chunk?
    n = int(np.ceil(len(layers) / chunks))

    # get chunks
    layer_sets = [layers[i: i + n] for i in range(0, len(layers), n)]

    return layer_sets[idx]


if __name__ == "__main__":

    logging.basicConfig(
        format='%(levelname)s:  %(message)s',
        level=logging.INFO  # Don't log DEBUG, but INFO, WARNING, ERROR and CRITICAL
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(accuracies.keys()),
        help="Which model to use. Supported values are: "
             f"{', '.join(list(accuracies.keys()))}.",
    )
    parser.add_argument(
        "--use_validation_set",
        action="store_true",
        help="Whether to use the validation set.",
    )
    parser.add_argument(
        "--n_chunks",
        type=int,
        required=True,
        help="How many chunks to create, see module docstring."
    )
    parser.add_argument(
        "--layer_chunk",
        type=int,
        required=True,
        help="Index of layer chunk to use, see module docstring.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="If the old method of storing to CSV should be used.",
    )
    # If activations should only be collected for the units we know we will need, pass units_file.
    parser.add_argument(
        "--units_file",
        type=str,
        default=None,
        required=False,
        help="Optionally: Path to units file that should be used to select layers."
    )

    args = parser.parse_args()

    # get the requested model
    target_model = load_model(args.model)

    # get units from file, extract all layers of interest
    if args.units_file is None:
        logging.info("Selecting all layers.")
        layers = get_relevant_layers(target_model, args.model)
    else:
        logging.info(f"Selecting layers as needed by {args.units_file}.")
        units = read_units_file(args.units_file)
        layers = get_layers_from_units_list(units)

    logging.info(f"Found {len(layers)} relevant layers: {layers}")

    # only choose one subset of this list of layers
    layers = get_subset(layers, args.n_chunks, args.layer_chunk)

    logging.info(f"Will collect activations for {len(layers)} layers: {layers}")

    main(target_model, args.model, layers, args.use_validation_set, args.csv)
