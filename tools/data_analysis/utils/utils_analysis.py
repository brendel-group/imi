"""
Utility functions for data analysis (statistical tests etc.)
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu, brunnermunzel
from tools.data_analysis.utils.utils_data import (
    load_and_parse_all_results,
    load_and_parse_trial_structure,
    append_trial_structure_to_results,
)

from typing import Dict, List, Optional


def get_all_units(df, split=True):
    """
    Obtains a list of all units in a df.

    :param df: the dataframe
    :param split: whether to split the units in layer and channel
    :returns: list[str] of all units that occur
    """
    units = []
    layers = df["layer"].unique()
    for layer in layers:
        ldf = df[df["layer"] == layer]
        layer_units = ldf["channel"].unique()
        if split:
            units.extend([(layer, unit) for unit in layer_units])
        else:
            units.extend(["__".join([layer, unit]) for unit in layer_units])
    return units


def get_correct_ratio_for_participant(df, pid):
    """
    Obtains the ratio of correct answers for df and participant-id.

    :param df: the dataframe of an experiment
    :param pid: the participant ID
    """
    pdf = df[df["participant_id"] == pid]

    correct_answers = pdf["correct"].values
    num_correct = np.sum(correct_answers)
    ratio_correct = num_correct / len(correct_answers)

    return ratio_correct


def get_correct_ratio_for_unit(df, layer, channel):
    """
    Obtains the ratio of correct answers for df and unit.

    :param df: the dataframe of an experiment
    :param layer: the layer of the unit
    :param channel: the channel of the unit
    """
    ldf = df[df["layer"] == layer]
    cdf = ldf[ldf["channel"] == channel]

    correct_answers = cdf["correct"].values
    num_correct = np.sum(correct_answers)
    ratio_correct = num_correct / len(correct_answers)

    return ratio_correct


def get_participant_performances(dfs):
    """
    Obtains the performances of all participants and returns them as a list.

    :param dfs: list of the dataframes to analyze
    :returns: list of length len(dfs), each element of which is a list of performances
    (e.g. [[0.8, 0.9, 0.85], [0.5, 0.6, 0.6]])
    """
    participant_performances = []
    for df in dfs:
        performances = []
        participant_ids = set(df["participant_id"].values)
        for pid in participant_ids:
            performances.append(get_correct_ratio_for_participant(df, pid))
        participant_performances.append(performances)
    return participant_performances


def get_unit_performances(dfs):
    """
    Obtains the performances of all units and returns them as a list.

    :param dfs: list of the dataframes to analyze
    :returns: list of length len(dfs), each element of which is a list of performances
    (e.g. [[0.8, 0.9, 0.85], [0.5, 0.6, 0.6]])
    """
    unit_performances = []
    for df in dfs:
        performances = []
        for layer, channel in get_all_units(df, True):
            performances.append(get_correct_ratio_for_unit(df, layer, channel))
        unit_performances.append(performances)
    return unit_performances


def get_exp_dfs(save_csv, results_folder, structure_path, fname):
    """
    Obtains the dataframes for main data and checks for an experiment.

    :param save_csv: whether to save the data to csv file or not
    :param results_folder: where to save results to
    :param structure_path: path to structure.json
    :param fname: filename for the csv file

    :returns: main df with data and df with check-results
    """
    df, df_checks = load_and_parse_all_results(results_folder)
    trial_structure = load_and_parse_trial_structure(structure_path)
    df = append_trial_structure_to_results(df, trial_structure)
    if save_csv:
        df.to_csv(os.path.join(results_folder, f"{fname}.csv"))
    return df, df_checks


def get_sub_dfs(df):
    """
    Given a dataframe of trial data, this function extracts the sub-dfs for normal,
    catch and demo trials.

    :param df: the dataframe filled with all data except for checks (first output of
    get_exp_df)

    :returns: main, catch and demo df
    """
    df_main = df[(df["catch_trial"] == False) & (df["is_demo"] == False)]
    df_catch_trials = df[(df["catch_trial"] == True) & (df["is_demo"] == False)]
    df_demo_trials = df[df["is_demo"] == True]
    return df_main, df_catch_trials, df_demo_trials


def run_kruskal_wallis(dfs, show_plots: bool = False):
    """
    Running a Kruskal-Wallis test (as replacement for ANOVA, because assumptions
    don't hold).

    :param dfs: list of dataframes in the conditions that are supposed to be compared.
    """

    # aggregate values over participants, to get one value for each unit
    performances = get_unit_performances(dfs)

    # to aggregate over units, which we don't do, use this:
    # performances = get_participant_performances(dfs)

    if show_plots:
        # first, plot histograms of data
        num_models = len(performances)
        cols = min(3, num_models)
        rows = int(np.ceil(num_models / cols))
        _fig, axs = plt.subplots(rows, cols, figsize=(15, 5))
        _fig.suptitle("Histograms of per-unit performances")

        axs = axs.flatten()

        for i, parts in enumerate(performances):

            ax = axs[i]

            epsilon = 0.001
            ax.hist(parts, bins=np.arange(-epsilon, 1.0 + epsilon, 0.1))

            ax.set_title(f"Model {i}")
            ax.set(
                xlabel=f"Proportion correct (mean={np.round(np.mean(parts), 3)}, "
                f"std={np.round(np.std(parts), 3)})",
                ylabel="Number of units",
            )

        plt.tight_layout()
        plt.show()

    statistic, pvalue = kruskal(*performances)
    print(
        f"Kruskal-Wallis test returned statistic {np.round(statistic, 3)} and "
        f"p-value {np.round(pvalue, 5)}"
    )

    if pvalue < 0.05:
        print("The difference between the distributions was statistically significant!")
    else:
        print("Unable to reject the null hypothesis of distributions being the same.")


def run_mwu_test(df_a, df_b, sig_level):
    """
    Runs the Mann-Whitney-U test between two dataframes (e.g. two conditions or
    two models).
    The performance values are aggregated per-unit.
    Only use this if the variance between the two groups is the same!
    https://www.tandfonline.com/doi/full/10.1080/00031305.2017.1305291

    :param df_a: the first dataframe
    :param df_b: the second df
    :param sig_level: the significance level at which to perform the test

    :returns: True if the test was significant, else otherwise
    """

    def get_group(df):
        """Turns a df into list of ratios."""
        units = set(zip(df["layer"], df["channel"]))
        ratios = []
        for layer, channel in units:
            ratios.append(get_correct_ratio_for_unit(df, layer, channel))
        return ratios

    # obtain per-unit performances
    group_a = get_group(df_a)
    group_b = get_group(df_b)

    std_a = np.std(group_a)
    std_b = np.std(group_b)

    # check if the variances are similar enough
    if np.abs(std_a**2 - std_b**2) > 0.01:
        print(
            "The variances of the two groups are too different to meaningfully "
            f"perform MWU test as a test of means! ({std_a**2} and {std_b**2})."
        )
        return False

    # run the test
    stat, pvalue = mannwhitneyu(group_a, group_b, axis=None)

    # interpret test result
    if pvalue < sig_level:
        print(
            f"MWU test was significant for alpha={sig_level}, at "
            f"p = {np.round(pvalue, 5)} with F = {np.round(stat, 3)}"
        )
    else:
        print(
            f"MWU test was NOT significant for alpha={sig_level}! "
            f"p = {np.round(pvalue, 5)}, F = {np.round(stat, 3)}"
        )

    return pvalue < sig_level


def run_brunner_munzel_test(df_a, df_b, sig_level):
    """
    Runs the Brunner-Munzel test between two dataframes (e.g. two conditions or
    two models).
    The performance values are aggregated per-unit.
    Use this if the variance between the two groups is different.
    https://journals.sagepub.com/doi/epub/10.1177/2515245921999602

    :param df_a: the first dataframe
    :param df_b: the second df
    :param sig_level: the significance level at which to perform the test

    :returns: True if the test was significant, else otherwise
    """

    def get_group(df):
        """Turns a df into list of ratios."""
        units = set(zip(df["layer"], df["channel"]))
        ratios = []
        for layer, channel in units:
            ratios.append(get_correct_ratio_for_unit(df, layer, channel))
        return ratios

    # obtain per-unit performances
    group_a = get_group(df_a)
    group_b = get_group(df_b)

    # obtain appropriate distribution
    dist = "t" if min(len(group_a), len(group_b)) <= 50 else "normal"

    # run test
    stat, pvalue = brunnermunzel(
        group_a, group_b, alternative="two-sided", distribution=dist
    )

    # interpret test
    if pvalue < sig_level:
        print(
            f"Brunner-Munzel test was significant for alpha={sig_level}, at "
            f"p = {np.round(pvalue, 5)} with W = {np.round(stat, 3)}"
        )
    else:
        print(
            f"Brunner-Munzel test was NOT significant for alpha={sig_level}! "
            f"p = {np.round(pvalue, 5)}, W = {np.round(stat, 3)}"
        )

    return pvalue < sig_level


def apply_all_checks(df_checks, results_folder: Optional[str] = None,
                     save_csv: bool = False):
    """
    Applies the quality assurance-checks (to check if participants were paying
    attention etc.)

    :param df_checks: dataframe for checks
    :param results_folder: the folder to which we store results
    :param save_csv: whether to save the checks-CSV
    """

    df_checks["instruction_time_details_extracted"] = df_checks.apply(
        lambda row: row["instruction_time_details"]["total_time"], axis=1
    )
    df_checks["total_response_time_details_extracted"] = df_checks.apply(
        lambda row: row["total_response_time_details"]["total_time"], axis=1
    )
    df_checks["row_variability_details_details_upper_extracted"] = df_checks.apply(
        lambda row: row["row_variability_details"]["n_upper_row"], axis=1
    )
    df_checks["row_variability_details_details_upper_extracted"] = df_checks.apply(
        lambda row: row["row_variability_details"]["n_upper_row"], axis=1
    )
    df_checks["row_variability_details_details_lower_extracted"] = df_checks.apply(
        lambda row: row["row_variability_details"]["n_lower_row"], axis=1
    )
    df_checks["catch_trials_details_ratio_extracted"] = df_checks.apply(
        lambda row: row["catch_trials_details"]["ratio"], axis=1
    )
    df_checks["catch_trials_details_correctly_answered_extracted"] = df_checks.apply(
        lambda row: row["catch_trials_details"]["correctly_answered"], axis=1
    )
    df_checks["demo_trials_details_extracted"] = df_checks.apply(
        lambda row: row["demo_trials_details"]["n_demo_trials_blocks"], axis=1
    )

    if save_csv:
        # save dataframes to csv
        df_checks.to_csv(os.path.join(results_folder, "df_exclusion_criteria.csv"))

    # Analyzing unique workers
    n_unique_tasks = df_checks.shape[0]
    n_unique_workers = len(df_checks["worker_id"].unique())
    print(f"We analyzed {n_unique_tasks} unique tasks")
    print(f"We had {n_unique_workers} unique workers")

    return df_checks


def analyze_participant_differences(df, figures_folder, fname, condition_name):
    """
    Comparing the participants of the pre-study.

    :param df: the main df
    :param figures_folder: the folder into which the figure should be stored
    :param fname: the filename to store at
    :param condition_name: the condition (only for plot title)
    """

    # get set of participant IDs
    participant_ids = set(df["participant_id"].values)
    print("Found participants: ", participant_ids)

    # dfs of natural/optimized stimuli
    std_df = df[df["model"].str.match("standard")]
    l2_df = df[df["model"].str.match("l2")]

    # figure out performance for each participant in each condition
    res = {}
    for pid in participant_ids:
        res[pid] = {}
        for name, data in [("standard", std_df), ("l2", l2_df)]:
            ratio = get_correct_ratio_for_participant(data, pid)
            res[pid][name] = ratio
            print(
                f"Participant {pid} gave {np.round(ratio*100,2)}% correct answers "
                f"for model {name}"
            )

    for model in ["standard", "l2"]:
        mean = np.mean([res[pid][model] for pid in participant_ids])
        std = np.std(
            ([res[pid][model] for pid in participant_ids]), ddof=1
        )  # estimating population sigma, so Bessel's correction
        print(f"Mean for model {model}: {mean} with std: {std}")

    # plot results
    model_colors = {"standard": "red", "l2": "blue"}

    plt.figure()
    plt.title(f"performances in condition {condition_name}")
    plt.xlabel("Participants (red: standard, blue: l2)")
    plt.ylabel("Proportion correct")
    for i, model in enumerate(["standard", "l2"]):
        data = [np.round(res[pid][model], 2) for pid in participant_ids]
        chart = plt.bar(
            list(np.arange(i * 0.4 - 0.2, len(participant_ids) - 0.2)),
            data,
            color=[model_colors[model]],
            width=0.4,
        )
        plt.bar_label(chart, labels=data)
    plt.xticks(list(range(len(participant_ids))))
    plt.ylim(0.45, 1.0)
    plt.savefig(os.path.join(figures_folder, f"{fname}.jpg"))
    plt.show()


def analyze_units(df):
    """
    Calculates per-unit mean accuracy and standard deviations in both conditions.

    :param df: the main df
    """

    # get set of units
    units = set(zip(df["layer"], df["channel"]))
    print(f"Found {len(units)} units.")

    # dfs of natural/optimized stimuli
    std_df = df[df["model"].str.match("standard")]
    l2_df = df[df["model"].str.match("l2")]

    # figure out performance for each participant in each condition
    res = {}
    for layer, channel in units:
        unit = "__".join([layer, channel])
        res[unit] = {}
        for name, data in [("standard", std_df), ("l2", l2_df)]:
            ratio = get_correct_ratio_for_unit(data, layer, channel)
            res[unit][name] = ratio

    for model in ["standard", "l2"]:
        per_unit_res = [
            res["__".join([layer, channel])][model] for layer, channel in units
        ]
        mean = np.mean(per_unit_res)
        std = np.std(
            per_unit_res, ddof=1
        )  # estimating population sigma, so Bessel's correction
        print(f"Mean for model {model}: {mean} with std: {std}")


def analyze_time(df, condition_name):
    """Calculates the average time needed for each trial."""

    # get set of participant IDs
    participant_ids = set(df["participant_id"].values)
    print("Found participants: ", participant_ids)

    # dfs of natural/optimized stimuli
    std_df = df[df["model"].str.match("standard")]
    l2_df = df[df["model"].str.match("l2")]

    def get_time(df, pid):
        """Get ratio of correct answers for df and participant-id"""
        pdf = df[df["participant_id"] == pid]
        times = pdf["rt"].values
        return np.mean(times) / 1000

    # figure out performance for each participant in each condition
    res = {}
    for pid in participant_ids:
        res[pid] = {}
        for name, data in [("standard", std_df), ("l2", l2_df)]:
            time = get_time(data, pid)
            res[pid][name] = time
            print(
                f"Participant {pid} took {np.round(time,2)} seconds on average "
                f"for model {name}"
            )

    for model in ["standard", "l2"]:
        mean = np.mean([res[pid][model] for pid in participant_ids])
        std = np.std(
            ([res[pid][model] for pid in participant_ids]), ddof=1
        )  # estimating population sigma, so Bessel's correction
        print(f"Mean for model {model}: {mean} with std: {std}")

    model_colors = {"standard": "red", "l2": "blue"}

    plt.figure()
    plt.title(f"performances in condition {condition_name}")
    plt.xlabel("Participants (red: standard, blue: l2)")
    plt.ylabel("average time (in seconds)")
    for i, model in enumerate(["standard", "l2"]):
        data = [np.round(res[pid][model], 2) for pid in participant_ids]
        chart = plt.bar(
            list(np.arange(i * 0.4 - 0.2, len(participant_ids) - 0.2)),
            data,
            color=[model_colors[model]],
            width=0.4,
        )
        plt.bar_label(chart, labels=data)
    plt.xticks(list(range(len(participant_ids))))
    plt.show()


def analyse_simple_experiment(
    dfs: Dict[str, Dict[str, str]],
    models: List[str],
    conditions: List[str],
    replicate=True,
):
    """
    Run the full analysis of the first, simple experiment that compares ResNet50
    and L2-robust ResNet50.
    Using alpha = 0.025 because of Bonferroni-correction: running two tests on
    each data block.

    :param dfs: dictionary that maps model and condition to a main-df (of real
        trials only)
    :param models: list of model names
    :param conditions: list of conditions
    :param replicate: True if we should also compare the natural and optimized
        condition for each model
    """

    # 1. run Kruskal-Wallis test for each condition to check if there were
    # significant differences
    for condition in conditions:
        print(
            f"Running Kruskal-Wallis across all models ({', '.join(models)}) "
            f"for condition {condition}:"
        )
        run_kruskal_wallis([dfs[model][condition] for model in models])
        print("----------")

        # 2. run MWU test between the two models in each condition
        print(
            f"Running Brunner-Munzel test between models 'standard' and 'L2' for "
            f"condition {condition}:"
        )
        run_brunner_munzel_test(dfs["standard"][condition], dfs["l2"][condition], 0.05)
        print("----------")

    if replicate:
        # 3. run MWU test for each model, to check if there was a difference between
        # the two conditions (replicating Borowski et al.)
        for model in models:
            print(
                "Running Brunner-Munzel test between conditions 'natural' and "
                f"'optimized' for model {model}:"
            )
            run_brunner_munzel_test(
                dfs[model]["natural"], dfs[model]["optimized"], 0.05
            )
            print("----------")
