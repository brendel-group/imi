import datetime

from spawn_experiment import get_verify_task_callback
from mturk import MTurkHIT
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file-name", required=True)
parser.add_argument(
    "--catch-trial-ratio-threshold",
    default=0,
    type=float,
    help="ratio of catch trials that must " "be passed",
)
parser.add_argument(
    "--min-total-response-time", default=0, type=float, help="time in [sec]"
)
parser.add_argument(
    "--max-total-response-time", default=-1, type=float, help="time in [sec]"
)
parser.add_argument(
    "--min-instruction-time", default=0, type=float, help="time in [sec]"
)
parser.add_argument(
    "--max-instruction-time", default=-1, type=float, help="time in [sec]"
)
parser.add_argument("--max-demo-trials-attempts", default=-1, type=float, help="")
parser.add_argument(
    "--row-variability-threshold",
    default=0,
    type=float,
    help="number of trials that must be "
    "chosen from the less frequently "
    "selected row",
)

args = parser.parse_args()

verify_task_callback = get_verify_task_callback(
    "2afc",
    args.catch_trial_ratio_threshold,
    args.min_total_response_time,
    args.max_total_response_time,
    args.min_instruction_time,
    args.max_instruction_time,
    args.row_variability_threshold,
    args.max_demo_trials_attempts,
)

# dummy data
hit = MTurkHIT(
    "1",
    "1",
    "1",
    "1",
    "1",
    1,
    datetime.datetime.now(),
    datetime.datetime.now(),
    1,
    "2afc",
)

with open(args.file_name, "r") as f:
    data = json.load(f)

assignment_data = data["Assignment"]
data, _ = hit._parse_assignment_response(assignment_data, verify_task_callback)
print(data["check_results"])
