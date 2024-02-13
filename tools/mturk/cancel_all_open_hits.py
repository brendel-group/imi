"""
Cancels all open HITS of a task.

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

import time
import warnings
import boto3
import botocore.exceptions

from aws_credentials import AWSCredentials
import datetime


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
            try:
                response = self._client.list_assignments_for_hit(
                    HITId=hit_id,
                    **kwargs,
                    AssignmentStatuses=["Submitted", "Approved", "Rejected"],
                )
            except botocore.exceptions.ClientError as ex:
                if "ThrottlingException" in str(ex):
                    time.sleep(5)
                    continue

            results += response["Assignments"]

            if "NextToken" in response:
                pagination_token = response["NextToken"]
            else:
                # retrieved all samples
                break

        return results


def cancel_open_hits(monitor):
    hits = monitor.list_hits()

    for hit in hits:
        if hit["HITStatus"] == "Assignable":
            response = monitor._client.update_expiration_for_hit(
                HITId=hit["HITId"],
                ExpireAt=datetime.datetime(year=2000, month=1, day=1),
            )
            if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
                print("Changed expiration date to now for HIT:", hit["HITId"])
            else:
                print("Failed to change expiration date to now for HIT:", hit["HITId"])

        if hit["HITStatus"] == "Reviewable":
            assignments = monitor.list_assignments_for_hit(hit["HITId"])

            for assignment in assignments:
                if (
                    assignment["AssignmentStatus"] == "Approved"
                    or assignment["AssignmentStatus"] == "Rejected"
                ):
                    continue
                response = monitor._client.approve_assignment(
                    AssignmentId=assignment["AssignmentId"]
                )
                if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
                    print(
                        f'Approved assignment {assignment["AssignmentId"]} for '
                        f'HIT {hit["HITId"]}'
                    )
                else:
                    print(
                        f'Could not approve assignment {assignment["AssignmentId"]} '
                        f'for HIT {hit["HITId"]}'
                    )


if __name__ == "__main__":
    while True:
        sandbox = input("Use sandbox or real? (accepted inputs: sandbox or real): ")
        if sandbox in ("sandbox", "real"):
            sandbox = sandbox == "sandbox"
            break
    print("Using environment:", "sandbox" if sandbox else "real")
    print()

    monitor = Monitor(sandbox=sandbox)

    cancel_open_hits(monitor)
