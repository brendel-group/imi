"""
We want to collect stimuli for the same units for the clip-resnet50 as for the other two resnets.
But the clip-resnet50 has a slightly different naming convention, so this script takes a path
to a units-file for a normal resnet50 and copies it to a target-location, renaming all units.
"""

import json
from argparse import ArgumentParser

from stimuli_generation.utils import read_units_file


def main(args):
    """Renames list of units."""

    units = read_units_file(args.src_file)

    units = [f'visual_{u}' for u in units]

    data = {"units": units}
    with open(args.tar_file, "w", encoding="utf-8") as fhandle:
        json.dump(data, fhandle)


if __name__ == "__main__":

    parser = ArgumentParser("Renaming units in src for clip.")
    parser.add_argument(
        "--src_file",
        "-s",
        type=str,
        required=True
    )
    parser.add_argument(
        "--tar_file",
        "-t",
        type=str,
        required=True
    )

    arguments = parser.parse_args()

    main(arguments)
