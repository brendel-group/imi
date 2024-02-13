"""
Simple script to join all of the json-files containing skipped units or units with
large alpha values.
"""
from argparse import ArgumentParser
from glob import glob

from stimuli_generation.utils import read_units_file, store_units


def main(args):
    """
    Reads all json files that match the pattern, joins the lists and writes them to
    the final file.

    :param args: CLI arguments
    """

    # find all files that match the pattern
    input_files = glob(args.pattern)

    # read them all and get the units
    units = []
    for file in input_files:
        units.extend(read_units_file(file))

    print(f"Found {len(units)} units!")

    # store the list of units as a new json file
    store_units(units, args.output)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--pattern", type=str, required=True, help="Pattern to match for the filename."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Name (without .json extension) of output-file.",
    )

    arguments = parser.parse_args()
    main(arguments)
