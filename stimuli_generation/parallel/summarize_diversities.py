"""
For each run over n GPUs, n files called 'diversity_i' are created, which contain the diversities.
I don't want to write to one file directly to avoid processes overwriting each other.
So to simplify plotting later, this script concatenates all diversity files.
"""

import os
import pandas as pd
from argparse import ArgumentParser


def main(args):

    final_df = pd.DataFrame()
    _, _, files = next(os.walk(args.source_dir))

    for file in files:
        if file.endswith(".csv") and "diversity_" in file:
            df = pd.read_csv(os.path.join(args.source_dir, file))
            final_df = pd.concat([final_df, df], ignore_index=True)

    final_df.to_csv(os.path.join(args.source_dir, "diversities.csv"), index=False)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--source_dir",
        "-s",
        type=str,
        required=True,
        help="Where to find diversity files"
    )
    args = parser.parse_args()

    main(args)
