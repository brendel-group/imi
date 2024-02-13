"""
Utility functions for loading and storing data.
"""

import os
import pickle
import datetime
import json

from glob import glob
import numpy as np
import pandas as pd

from tools.mturk.mturk import RepeatedTaskResult

from collections import namedtuple

# for some reason, pickle doesn't recognize the RTR if I don't tell it explicitly
class CustomUnpickler(pickle.Unpickler):
    """Custom Unpickler for RepeatedTaskResults."""

    def find_class(self, module, name):
        if name == "RepeatedTaskResult":
            return RepeatedTaskResult
        return super().find_class(module, name)


DatedFileNameRepeatedTaskResult = namedtuple(
    "RepeatedTaskResult",
    [
        "task_id",
        "raw_responses",
        "responses",
        "approved_responses",
        "rejected_responses",
        "creation_time",
        "file_name"
    ],
)


def load_results(data_folder):
    """
    Loads experiment results as pickled RepeatedTaskResult object.mro

    :param data_folder: path to folder containing all task-results as pkl files
    :returns: list[RepeatedTaskResult]
    """
    result_fns = glob(os.path.join(data_folder, "result_task_*.pkl"))

    all_results = []

    for result_fn in result_fns:
        with open(result_fn, "rb") as f:
            result = CustomUnpickler(f).load()

        modification_timestamp = os.path.getmtime(result_fn)
        modification_time = datetime.datetime.fromtimestamp(modification_timestamp)

        if len(result) == 1:
            for i in range(len(result)):
                result[i] = DatedFileNameRepeatedTaskResult(*result[i], creation_time = modification_time, file_name=result_fn)
            all_results += result
        else:
            result = DatedFileNameRepeatedTaskResult(*result, creation_time = modification_time, file_name=result_fn)
            all_results.append(result)

    return all_results


def parse_results(tasks, use_raw_data: bool = False):
    """
    Converts list of RepeatedTaskResult objects to pandas dataframe.

    :param tasks: list[RepeatedTaskResult]
    :param use_raw_data: if True, will use raw data ("raw_data") from the task,
        otherwise will use the processed main data ("main_data")
    :returns: dataframe
    """

    dfs = []
    for _i_task, task_data in enumerate(tasks):
        dfs_per_task = []

        for response_idx, response in enumerate(task_data.raw_responses):
            # If you want to look at the demo trials and other raw data,
            # load pd.DataFrame(response_data["raw_data"])
            if use_raw_data:
                response_data = response["raw_data"]
                response_data = [it for it in response_data if it[
                    "trial_type"].endswith("-image-confidence-response")]
            else:
                response_data = response["main_data"]
            if response_data is None:
                # Ignore empty responses (which can rarely happen if a HIT/Assignment times out)
                continue

            response_df = pd.DataFrame(response_data)
            response_df["response_index"] = response_idx

            response_df["participant_id"] = response["worker_id"]

            # For each response, subtract the index of the first non-demo trial.
            response_df[
                "corrected_trial_index"] = response_df.trial_index - response_df[
                ~response_df["is_demo"]].trial_index.min()

            dfs_per_task.append(response_df)

        task_df = pd.concat(dfs_per_task, axis=0)

        task_df["task_id"] = task_data.task_id
        task_df["task_number"] = int(task_data.task_id.split("-")[-1])
        task_df["result_creation_time"] = task_data.creation_time
        task_df["result_file_name"] = task_data.file_name

        dfs.append(task_df)

    df = pd.concat(dfs, axis=0)
    df = df.reset_index().drop("index", axis=1)

    return df


def parse_check_results(tasks):
    """
    Creates a dataframe of check-results in which all necessary checks are conducted.

    :param tasks: list[RepeatedTaskResult], the results for all tasks
    """
    df = []
    for task in tasks:
        for response_idx, response in enumerate(task.raw_responses):
            check_results = response["check_results"]
            if check_results is None:
                # Ignore empty responses (which can rarely happen if a HIT/Assignment times out).
                continue

            df.append(
                {
                    "task_id": task.task_id,
                    "response_index": response_idx,
                    "passed_checks": response["passed_checks"],
                    "worker_id": response["worker_id"],
                    **{f"{k}_result": check_results[k][0] for k in check_results},
                    **{f"{k}_details": check_results[k][1] for k in check_results},
                })

    df = pd.DataFrame(df)
    return df


def load_and_parse_all_results(base_folder):
    """
    Loads all .pkl files from the target directory and produces two dataframes, one
    with the results, one with the checks.

    :param base_folder: str, the folder that contains .pkl files

    :returns: two dataframes (see above)
    """
    results = load_results(base_folder)

    assert (
        len(results) > 0
    ), "No results (as .pkl files) found at location {base_folder}!"

    df_checks = parse_check_results(results)

    df = parse_results(results)

    return df, df_checks


def load_and_parse_trial_structure(structure_path):
    """
    Loads the .json-file with trial structure.

    :param structure_path: path to trial structure json file
    :returns: a dict representing the trial structure
    """

    def parse_trials_structure(trials):
        results = []
        for trial in trials:
            query_path = trial["queries"]
            parts = query_path.split("/")
            model = parts[-5]
            layer = parts[-4]
            channel = parts[-3].split("_")[-1]
            batch = parts[-1].split("_")[-1]
            mode = trial["mode"]

            if mode == "catch_trial":
                # Ignore model information for catch trials.
                model = "catch/manual"
                batch = -1
                layer = parts[-3]
                channel = parts[-2].split("_")[-1]

            results.append(
                dict(batch=batch, channel=channel, layer=layer, mode=mode, model=model,
                     query_path=query_path)
            )
        return results

    with open(structure_path, "r", encoding="utf-8") as f:
        raw_structure = json.load(f)

    structure = {}
    for item in raw_structure["tasks"]:
        structure[item["index"]] = {
            k: parse_trials_structure(item[k]) for k in item if k != "index"
        }

    return structure


def append_trial_structure_to_results(df, structure):
    """
    Reads the json file with information on the trials and attaches the info contained
    in them as extra columns (batch, channel, layer, mode) to the dataframe.

    :param df: the dataframe containing experiment results
    :param structure: dict containing the structure of trials
    """
    df = df.copy(deep=True)

    # Only append structure to non-demo trials.
    demo_df = df[df.is_demo].copy(deep=True)
    non_demo_df = df[~df.is_demo].copy(deep=True)

    # merge structure with df
    batch_column = []
    channel_column = []
    layer_column = []
    model_column = []
    mode_column = []
    query_path_column = []
    for i in non_demo_df.index:
        task_number = non_demo_df.loc[i, "task_number"]
        trial_number = non_demo_df.loc[i, "corrected_trial_index"]
        info = structure[task_number]["trials"][trial_number]
        batch_column.append(info["batch"])
        channel_column.append(info["channel"])
        layer_column.append(info["layer"])
        mode_column.append(info["mode"])
        model_column.append(info["model"])
        query_path_column.append(info["query_path"])

    non_demo_df["batch"] = batch_column
    non_demo_df["channel"] = channel_column
    non_demo_df["layer"] = layer_column
    non_demo_df["mode"] = mode_column
    non_demo_df["model"] = model_column
    non_demo_df["query_path"] = query_path_column

    df = pd.concat([demo_df, non_demo_df], axis=0)

    return df



def append_checks_to_results(df_results, df_checks):
    df_results_checks = df_results.copy(deep=True)
    
    columns_to_append = []
    for c in df_checks.columns:
        if c not in ("task_id", "participant_id", "response_index"):
            if c in df_results.columns:
                raise ValueError(f"Cannot append column {c} as it already exists.")
            columns_to_append.append(c)

    for c in columns_to_append:
        df_results_checks[c] = None
        
        for _, row in df_checks.iterrows():
            mask = (
                    df_results_checks["participant_id"] == row["worker_id"]) & (
                    df_results_checks["task_id"] == row["task_id"]) & (
                    df_results_checks["response_index"] == row["response_index"])
            try:
                assert mask.sum() > 0
                df_results_checks.loc[mask, c] = pd.Series(
                    [row[c] for _ in range(mask.sum())],
                    index=df_results_checks.index[mask])
            except Exception as ex:
                print(mask.sum(), row[c], c)
                raise ex

    df_results_checks = df_results_checks.infer_objects()
        
    return df_results_checks


def update_worker_names(df, name_map):
    """
    Replaces the participant ID in a dataframe with the name given by name_map,
    which maps the task_number to the new worker name.

    :param df: the dataframe
    :param name_map: dict that maps task_number to new worker-name

    :returns: the updated df
    """
    df["participant_id"] = df.apply(lambda row: name_map[row["task_number"]], axis=1)
    return df
