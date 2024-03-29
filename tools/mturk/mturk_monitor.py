"""

From Human Adversarials project.

Copyright (c) 2020-2021 Roland S. Zimmermann and Judy Borowski

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Copyright (c) 2024 Roland S. Zimmermann and Thomas Klein

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import argparse
import logging
import warnings
import boto3
import botocore.exceptions

from aws_credentials import AWSCredentials
import pandas as pd
import time
import datetime
from dateutil.tz import tzlocal


class Monitor:
    def __init__(self, sandbox=True):
        if sandbox:
            warnings.warn("Using MTurk Sandbox environment.")
            self._endpoint_url = (
                "https://mturk-requester-sandbox.us-east-1.amazonaws.com"
            )
        else:
            warnings.warn("Using real MTurk environment.")
            self._endpoint_url = "https://mturk-requester.us-east-1.amazonaws.com"

        self.sandbox = sandbox

        self._setup_client()

    def _setup_client(self):
        region_name = "us-east-1"

        self._client = boto3.client(
            "mturk",
            endpoint_url=self._endpoint_url,
            region_name=region_name,
            aws_access_key_id=AWSCredentials.aws_access_key_id(),
            aws_secret_access_key=AWSCredentials.aws_secret_access_key(),
        )

    def list_hits(self):
        results = []
        pagination_token = None
        while True:
            if pagination_token is not None:
                kwargs = dict(NextToken=pagination_token)
            else:
                kwargs = {}
            response = self._client.list_hits(**kwargs, MaxResults=100)

            results += response["HITs"]

            if "NextToken" in response:
                pagination_token = response["NextToken"]
            else:
                # retrieved all samples
                break

        return results

    def list_assignments_for_hit(self, hit_id):
        results = []
        pagination_token = None
        while True:
            if pagination_token is not None:
                kwargs = dict(NextToken=pagination_token)
            else:
                kwargs = {}
            response = self._client.list_assignments_for_hit(
                HITId=hit_id,
                **kwargs,
                AssignmentStatuses=["Submitted", "Approved", "Rejected"],
            )

            results += response["Assignments"]

            if "NextToken" in response:
                pagination_token = response["NextToken"]
            else:
                # retrieved all samples
                break

        return results


def collect_data(monitor):
    hits = monitor.list_hits()

    subject_table = []
    for hit in hits:
        assignments = monitor.list_assignments_for_hit(hit["HITId"])

        for asgn in assignments:
            subject = dict(
                AssignmentId=asgn["AssignmentId"],
                WorkerId=asgn["WorkerId"],
                HITId=asgn["HITId"],
                AssignmentStatus=asgn["AssignmentStatus"],
                AcceptTime=asgn["AcceptTime"],
                ApprovalTime=asgn["ApprovalTime"]
                if asgn["AssignmentStatus"] == "Approved"
                else None,
                Reward=hit["Reward"]
                if asgn["AssignmentStatus"] == "Approved"
                else "0.00",
            )
            subject_table.append(subject)

    if len(subject_table) > 0:
        subject_table = pd.DataFrame(subject_table)
    else:
        # if no data exists, create some required columns
        subject_table = pd.DataFrame(
            columns=["ApprovalTime", "AcceptTime", "AssignmentId"]
        )

    subject_table["Duration"] = (
        subject_table["ApprovalTime"] - subject_table["AcceptTime"]
    )
    subject_table = subject_table.set_index("AssignmentId")
    hit_table = pd.DataFrame(hits)

    if len(hit_table.columns) == 0:
        hit_table = None
    else:
        hit_table = hit_table.set_index("HITId")

    return hits, (subject_table, hit_table)


def refresh_data(monitor, hits, subject_table, hits_table):
    def load_details_for_hit(hit):
        nonlocal subject_table
        assignments = monitor.list_assignments_for_hit(hit["HITId"])

        for asgn in assignments:
            subject = dict(
                AssignmentId=asgn["AssignmentId"],
                WorkerId=asgn["WorkerId"],
                HITId=asgn["HITId"],
                AssignmentStatus=asgn["AssignmentStatus"],
                AcceptTime=asgn["AcceptTime"],
                ApprovalTime=asgn["ApprovalTime"]
                if asgn["AssignmentStatus"] == "Approved"
                else None,
                Reward=hit["Reward"]
                if asgn["AssignmentStatus"] == "Approved"
                else "0.00",
            )

            # if asgn["AssignmentId"] in subject_table.index:
            # entry already exists
            # just update it
            new_subject = {**subject}
            del new_subject["AssignmentId"]
            new_subject_series = pd.Series(new_subject)
            subject_table.loc[
                asgn["AssignmentId"], new_subject_series.index
            ] = new_subject_series
            # else:
            #    # add new entry
            #    subject_table = subject_table.append(
            #        pd.Series(subject, index="AssignmentId")
            #    )

    current_hits = monitor.list_hits()

    # refresh details for unaccepted HITs
    for hit in [
        h
        for h in hits
        if h["NumberOfAssignmentsPending"] != 0
        or h["NumberOfAssignmentsAvailable"] != 0
    ]:
        load_details_for_hit(hit)

        # Check if we previously already had HITs with the same ID; if so,
        # check that there only exists exactly 1 recording, otherwise, warn the user.
        potentially_new_hit_data = [
            h for h in current_hits if h["HITId"] == hit["HITId"]
        ]
        if len(potentially_new_hit_data) == 0:
            continue
        else:
            if len(potentially_new_hit_data) > 1:
                logging.warning(
                    "Detected multiple recordings with the same HIT. "
                    "Please restart monitor to potentially fix this inconsistency."
                )
        new_hit_data = potentially_new_hit_data[0]
        # drop HITId index as this has not changed; for this, we need to create a copy
        new_hit_data = {**new_hit_data}
        del new_hit_data["HITId"]
        new_hit_data_series = pd.Series(new_hit_data)
        hit_table.loc[hit["HITId"], new_hit_data_series.index] = new_hit_data_series

    old_hit_ids = [h["HITId"] for h in hits]
    new_hits = [h for h in current_hits if h["HITId"] not in old_hit_ids]

    hits += new_hits

    for hit in new_hits:
        load_details_for_hit(hit)

        new_hit_data = {**hit}
        del new_hit_data["HITId"]
        new_hit_data_series = pd.Series(new_hit_data)
        hit_table.loc[hit["HITId"], new_hit_data_series.index] = new_hit_data_series

    subject_table = subject_table.sort_values(by="AcceptTime")
    subject_table["Duration"] = (
        subject_table["ApprovalTime"] - subject_table["AcceptTime"]
    )

    return hits, (subject_table, hits_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", choices=("real", "sandbox"), required=True)
    args = parser.parse_args()
    monitor = Monitor(sandbox=args.environment == "sandbox")

    pd.options.display.min_rows = 30
    pd.set_option("display.max_rows", 100)

    def display_data(subject_table, hit_table):
        open_hit_ids = hit_table[
            (hit_table["NumberOfAssignmentsAvailable"] > 0)
            & (hit_table["HITStatus"] != "Unassignable")
            & (hit_table["HITStatus"] != "Reviewable")
            & (hit_table["Expiration"] > datetime.datetime.now(tz=tzlocal()))
        ].index.tolist()
        open_but_accepted_hit_ids = hit_table[
            hit_table["NumberOfAssignmentsPending"] > 0
        ].index.tolist()
        closed_hit_ids = hit_table[
            hit_table["HITStatus"] == "Reviewable"
        ].index.tolist()
        n_open_hits = len(open_hit_ids)
        n_open_but_accepted_hits = len(open_but_accepted_hit_ids)
        n_closed_hits = len(closed_hit_ids)

        print(datetime.datetime.now())
        print(
            f"Open HITs ({n_open_hits}): "
            + ", ".join([it[:3] + "..." + it[-3:] for it in open_hit_ids])
        )
        print(
            f"HITs accepted by workers ({n_open_but_accepted_hits}): "
            + ", ".join([it[:3] + "..." + it[-3:] for it in open_but_accepted_hit_ids])
        )
        print(f"Closed HITs: {n_closed_hits}")
        if len(subject_table) == 0:
            print("No records of subjects found.")
        else:
            subject_table = subject_table.copy(deep=True)
            subject_table = subject_table.sort_values(by="AcceptTime")
            print(subject_table.tail(pd.options.display.min_rows))
        print("")

    while True:
        try:
            hits, (subject_table, hit_table) = collect_data(monitor)
            break
        except botocore.exceptions.ClientError:
            pass

    if subject_table is not None and hit_table is not None:
        display_data(subject_table, hit_table)
    else:
        print("No data found yet.")

    while True:
        time.sleep(60)
        try:
            if subject_table is None or hit_table is None:
                hits, (subject_table, hit_table) = collect_data(monitor)
            else:
                hits, (subject_table, hit_table) = refresh_data(
                    monitor, hits, subject_table, hit_table
                )
            if subject_table is not None and hit_table is not None:
                display_data(subject_table, hit_table)
            else:
                print("No data found yet.")
        except botocore.exceptions.ClientError:
            pass
