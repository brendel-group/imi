"""
Extracts information on all qualification ids we assigned so far.

This script first looks for all qualification types that match a given pattern,
and then retrieves all workers that have this qualification.
This is then stored as a pickle file.

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
import pickle
import time
import warnings
import boto3
import botocore.exceptions

from aws_credentials import AWSCredentials
import datetime
import logging
import argparse


class Client:
    def __init__(self, sandbox=False):
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

    @property
    def __logger(self):
        return logging.getLogger("mturk")

    def _setup_client(self):
        region_name = "us-east-1"

        self._client = boto3.client(
            "mturk",
            endpoint_url=self._endpoint_url,
            region_name=region_name,
            aws_access_key_id=AWSCredentials.aws_access_key_id(),
            aws_secret_access_key=AWSCredentials.aws_secret_access_key(),
        )

    def list_workers_with_qualification_type(self, qualification_type_id: str):
        results = []
        pagination_token = None
        while True:
            if pagination_token is not None:
                kwargs = dict(NextToken=pagination_token)
            else:
                kwargs = {}
            response = self._client.list_workers_with_qualification_type(
                QualificationTypeId=qualification_type_id, **kwargs, MaxResults=100
            )

            results += response["Qualifications"]

            if "NextToken" in response:
                pagination_token = response["NextToken"]
            else:
                # retrieved all samples
                break

        return results

    def get_previous_single_participation_qualifications(
        self, qualification_pattern: str = ""
    ):
        self.__logger.info(
            f"Looking for qualification types matching the pattern `{qualification_pattern}`"
        )
        qualifications = []
        pagination_token = None
        while True:
            if pagination_token is not None:
                kwargs = dict(NextToken=pagination_token)
            else:
                kwargs = {}
            try:
                response = self._client.list_qualification_types(
                    Query=qualification_pattern,
                    MustBeOwnedByCaller=True,
                    MustBeRequestable=False,
                    **kwargs, MaxResults=100
                )
            except:
                time.sleep(1)
                continue

            qualifications += response["QualificationTypes"]

            if "NextToken" in response:
                pagination_token = response["NextToken"]
            else:
                # retrieved all samples
                break

        return qualifications


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sandbox", action="store_true")
    parser.add_argument("--qualifications-pattern", type=str, required=True)

    args = parser.parse_args()

    client = Client(sandbox=args.sandbox)

    qualifications = client.get_previous_single_participation_qualifications(
        args.qualifications_pattern)

    qualification_workers = dict()
    for q in qualifications:
        qualification_workers[q["QualificationTypeId"]
        ] = client.list_workers_with_qualification_type(q["QualificationTypeId"])

    with open("qualification_workers.pkl", "wb") as f:
        pickle.dump((qualifications, qualification_workers), f)

if __name__ == "__main__":
    main()
