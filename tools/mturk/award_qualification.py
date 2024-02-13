"""
Awards a summary qualification id to a group of workers that have some other
qualification.

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

            for it in response["Qualifications"]:
                if it["Status"] == "Granted":
                    results.append(it["WorkerId"])

            if "NextToken" in response:
                pagination_token = response["NextToken"]
            else:
                # retrieved all samples
                break

        return results

    def get_previous_single_participation_qualification_ids(
        self, previous_single_participation_qualifications
    ):
        """For each of the previous qualification names, get the internal ID
        needed to assign the qualification to workers"""

        ids = []
        for qualification in previous_single_participation_qualifications:
            try:
                self.__logger.info(
                    "Getting ID for qualification type "
                    "`{0}` to restrict participation".format(qualification)
                )
                response = self._client.list_qualification_types(
                    Query="brendellab-ic-" + qualification, MustBeRequestable=False,
                    MustBeOwnedByCaller=True
                )

                # Since the API returns all qualification types that contain the
                # query string, we need to filter out the ones that don't match
                # exactly.
                response["QualificationTypes"] = [
                    qt for qt in response["QualificationTypes"]
                    if qt["Name"] == "brendellab-ic-" + qualification]

                for item in response["QualificationTypes"]:
                    ids.append(item["QualificationTypeId"])

                if len(response["QualificationTypes"]) == 1:
                    self.__logger.info(
                        "Successfully got ID {0} for qualification type {1}".format(
                            ids[-1], qualification
                        )
                    )
                else:
                    raise ValueError(
                        "Could not find qualification ID for "
                        f"qualification `{qualification}`"
                    )
            except Exception as ex:
                self.__logger.warning(
                    f"Failed to get ID qualification type `{qualification}`: {repr(ex)}"
                )

        return ids

    def setup_single_participation_qualification(
        self, single_participation_qualification: str
    ):
        """Create a qualification type to assign later to workers"""

        try:
            self.__logger.info(
                "Creating qualification type `{0}` to restrict participation".format(
                    single_participation_qualification
                )
            )

            response = self._client.create_qualification_type(
                Name="brendellab-ic-" + single_participation_qualification,
                Description="This qualification indicates that a worker has"
                "participated in a recent study executed by the Brendellab"
                "with id " + single_participation_qualification,
                QualificationTypeStatus="Active",
            )
            single_participation_qualification_id = response["QualificationType"][
                "QualificationTypeId"
            ]
            self.__logger.info(
                "Successfully created qualification type `{0}` with ID `{1}`".format(
                    single_participation_qualification,
                    single_participation_qualification_id,
                )
            )
        except Exception as ex:
            self.__logger.warning("Failed to create qualification type: " + repr(ex))
            self.__logger.info("Trying to treat it as a previous qualification type...")
            single_participation_qualification_id = (
                self.get_previous_single_participation_qualification_ids(
                    [single_participation_qualification]
                )[0]
            )

        if single_participation_qualification_id is None:
            self.__logger.warning("No single participation qualification ID used")
        else:
            self.__logger.info(
                "Using single participation qualification ID "
                f"`{single_participation_qualification_id}`"
            )

        return single_participation_qualification_id

    def award_participation_qualification(
        self, worker_id: str, single_participation_qualification_id: str
    ):
        """Awards a qualification to an MTurk worker"""

        while True:
            try:
                self.__logger.info(
                    "Trying to assign participation qualification to worker: "
                    + worker_id
                )

                self._client.associate_qualification_with_worker(
                    QualificationTypeId=single_participation_qualification_id,
                    WorkerId=worker_id,
                    SendNotification=False,
                    IntegerValue=1,
                )

                self.__logger.info(
                    "Assigned participation qualification to worker: " + worker_id
                )

                return True
            except Exception as ex:
                self.__logger.info("Failed to award qualification type: " + repr(ex))
                time.sleep(5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sandbox", action="store_true")
    parser.add_argument("--previous-qualification", type=str, required=True)
    parser.add_argument("--new-qualification", type=str, required=True)

    args = parser.parse_args()

    client = Client(sandbox=args.sandbox)

    previous_qualification_id = (
        client.get_previous_single_participation_qualification_ids(
            [args.previous_qualification]
        )[0]
    )
    workers = client.list_workers_with_qualification_type(previous_qualification_id)
    new_qualification_id = client.setup_single_participation_qualification(
        args.new_qualification
    )
    print("Previous qualification ID:", previous_qualification_id)
    print("New qualification ID:", new_qualification_id)
    print("Number of affected workers:", len(workers))
    print("Awarding qualification...")

    for w in workers:
        client.award_participation_qualification(w, new_qualification_id)


if __name__ == "__main__":
    main()
