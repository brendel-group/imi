"""
Script to sample the neurons across the network that should be chosen available
for experiments. The units are written to a json-file, which can then be read by
get_optimized_stimuli.py and extract_exemplars.py.
"""

import json
from argparse import ArgumentParser

import torch
import timm
import open_clip
import numpy as np

from stimuli_generation.utils import (
    load_model,
    get_model_layers,
    test_layer_relevance,
    read_units_file,
    accuracies,
)

# Maps a model-type to a dictionary that maps layer_name to number of output units of
# that layer
layer_sizes = {
    "resnet50": None,
    "inception": None,
    "wide_resnet": None,
    "densenet": None,
    "convnext": None,
    "vit_b32": None
}

# maps a model name to its type of model
model_types = {
    "resnet50-l2": "resnet50",
    "resnet50-linf": "resnet50",
    "resnet50": "resnet50",
    "clip-resnet50": "resnet50",
    "wide_resnet50": "wide_resnet",
    "densenet_201": "densenet",
    "googlenet": "inception",
    "convnext_b": "convnext",
    "clip-vit_b32": "vit_b32",
    "in1k-vit_b32": "vit_b32",
}


def init_layer_sizes(model_name, model):
    """
    Initializes the layer_sizes dictionary so that it maps the key to a dictionary that
    maps layer names to the number of output units of that layer.

    :param model_name: the name of the model
    :param model: the actual pytorch model to source layer sizes from
    """

    last_seen_size = 0

    def get_size(module):
        """Returns the output size of a module."""
        nonlocal last_seen_size
        size = 0
        if isinstance(module, torch.nn.Conv2d):
            size = module.out_channels
        elif isinstance(module, torch.nn.BatchNorm2d):
            size = module.num_features
        elif isinstance(module, timm.models.convnext.ConvNeXtBlock):
            size = module.conv_dw.out_channels
        elif (isinstance(module, 
                         (timm.models.layers.LayerNorm,
                          timm.models.layers.LayerNorm2d,
                          open_clip.transformer.LayerNorm,
                          torch.nn.modules.normalization.LayerNorm))):
            size = module.normalized_shape[0]
        elif isinstance(module, torch.nn.modules.linear.Linear):
            size = module.out_features
        elif isinstance(module, torch.nn.Identity):  # shortcuts
            # This is a bit hacky: because this layer does not know its own size, we
            # just take the size of the layer before.
            # NOTE: this assumes that layers are returned in correct order by
            # get_model_layers.
            return last_seen_size
        else:
            raise ValueError(f"Module type not understood: {type(module)}")

        # remember the last seen size, so that shortcut units know their size
        last_seen_size = size

        return size

    layers = {
        layer: get_size(module)
        for layer, module in get_model_layers(model, True).items()
        if test_layer_relevance(layer, model_name)
    }

    layer_sizes[model_types[model_name]] = layers


def main(args, blacklist):
    """
    Samples n_units units for model_name.

    :param args: the CLI arguments
    :param blacklist: list of units that should be avoided
    """

    np.random.seed(args.seed)

    # Load the model and populate layer_sizes-dictionary to know possible number of
    # units per layer.
    model = load_model(args.model_name)

    if "clip" in args.model_name:
        model = model.visual

    init_layer_sizes(args.model_name, model)

    # sample a layer, then sample a unit from that layer
    model_layer_sizes = layer_sizes[model_types[args.model_name]]
    possible_layers, weights = zip(*model_layer_sizes.items())
    weights /= np.sum(weights)

    units = []
    for _ in range(args.n_units):

        while True:

            # 1. select a layer
            layer = np.random.choice(possible_layers, size=1)[
                0
            ]  # p=weights to weigh by layer size

            # 2. select a unit from that layer
            possible_units = model_layer_sizes[layer]
            unit = np.random.randint(0, possible_units)
            unit_name = "__".join([layer, str(unit)])

            if "clip" in args.model_name:
                unit_name = f"visual_{unit_name}"

            # 3. repeat if unit was already chosen
            if unit_name not in units and (
                blacklist is None or unit_name not in blacklist
            ):
                units.append(unit_name)
                break

    # store the chosen units
    data = {"units": units}
    with open(args.filename, "w", encoding="utf-8") as f:
        json.dump(data, f)

    print(f"Sampled {args.n_units} neurons.")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--model_name",
        choices=list(accuracies.keys()),
        required=True,
        help="Which model to use. Supported values are: "
             f"{', '.join(list(accuracies.keys()))}.",
    )
    parser.add_argument(
        "--n_units", type=int, required=True, help="How many units to sample"
    )
    parser.add_argument(
        "--filename",
        type=str,
        required=True,
        help="Filename of the json file to which results are written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=42,
        help="Random seed for pseudorandom numbers.",
    )
    parser.add_argument(
        "--blacklist",
        nargs="+",
        type=str,
        required=False,
        help="Files to avoid taking units from.",
    )

    arguments = parser.parse_args()

    blacklist = None
    if arguments.blacklist:
        blacklist = []
        for blist in arguments.blacklist:
            blacklist.extend(read_units_file(blist))

    main(arguments, blacklist)
