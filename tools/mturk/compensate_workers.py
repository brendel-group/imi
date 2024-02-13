"""
In order to compensate workers who experienced issues, we need to assign them a
new qualifier so that only they can view the special compensation HIT.

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

import warnings
import boto3
import time

from aws_credentials import AWSCredentials
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

    def get_ids_for_existing_qualifications(
        self, qualifications
    ):
        """
        Takes a list of qualifications and returns their IDs.
        """

        ids = []
        for qualification in qualifications:
            try:
                self.__logger.info(
                    "Getting ID for qualification type "
                    "`{0}`.".format(qualification)
                )
                response = self._client.list_qualification_types(
                    Query="brendellab-ic-" + qualification,
                    MustBeRequestable=False
                )

                relevant_qualifications = [
                    qt for qt in response["QualificationTypes"]
                    if qt["Name"] == "brendellab-ic-" + qualification
                ]

                for item in relevant_qualifications:
                    ids.append(item["QualificationTypeId"])

                if len(relevant_qualifications) == 1:
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


    def setup_compensation_qualification(
        self, compensation_qualification: str
    ):
        """Create a qualification type to assign later to workers"""

        try:
            self.__logger.info(
                "Creating qualification type `{0}` to mark workers for compensation.".format(
                    compensation_qualification
                )
            )

            response = self._client.create_qualification_type(
                Name="brendellab-ic-" + compensation_qualification,
                Description="This qualification indicates that a worker has"
                "participated in a recent study executed by the Brendellab"
                "that had technical issues so the worker did not get paid.",
                QualificationTypeStatus="Active",
            )
            compensation_qualification_id = response["QualificationType"][
                "QualificationTypeId"
            ]
            self.__logger.info(
                "Successfully created qualification type `{0}` with ID `{1}`".format(
                    compensation_qualification,
                    compensation_qualification_id,
                )
            )
        except Exception as ex:
            self.__logger.warning("Failed to create qualification type: " + repr(ex))
            self.__logger.info("Checking if qualification type already exists...")
            compensation_qualification_id = (
                self.get_ids_for_existing_qualifications(
                    [compensation_qualification]
                )[0]
            )

        if compensation_qualification_id is None:
            self.__logger.warning("No qualification ID found!")
        else:
            self.__logger.info(
                "Using qualification ID "
                f"`{compensation_qualification_id}`"
            )

        return compensation_qualification_id


    def award_compensation_qualification(
        self, worker_id: str, compensation_qualification_id: str
    ):
        """Awards a qualification to an MTurk worker"""

        while True:
            try:
                self.__logger.info(
                    "Trying to assign compensation qualification to worker: "
                    + worker_id
                )

                self._client.associate_qualification_with_worker(
                    QualificationTypeId=compensation_qualification_id,
                    WorkerId=worker_id,
                    SendNotification=False,
                    IntegerValue=1,
                )

                self.__logger.info(
                    "Assigned compensation qualification to worker: " + worker_id
                )

                return True

            except Exception as ex:
                self.__logger.info("Failed to award qualification type: " + repr(ex))
                time.sleep(5)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sandbox", "-s", action="store_true")
    parser.add_argument("--workerID-file", "-widf", type=str, required=True, help="Path to csv file with worker IDs")
    parser.add_argument("--new-qualification", type=str, required=True)

    args = parser.parse_args()

    client = Client(sandbox=args.sandbox)

    with open(args.workerID_file, "r") as fhandle:
        worker_ids = fhandle.readlines()
    worker_ids = [id.strip() for id in worker_ids]
    worker_ids = set(worker_ids)

    print("Workers to be reimbursed:", worker_ids)

    if (
        input(f"Will award compensation qualification {args.new_qualification} to {len(worker_ids)} workers. Type 'yes' to continue:")
        != "yes"
    ):
        print("Abort")
        return

    # get ID of new qualification
    qid = client.setup_compensation_qualification(args.new_qualification)

    assert qid is not None, "Could not find Qualification Type ID! Aborting..."

    for workerID in worker_ids:
        client.award_compensation_qualification(workerID, str(qid))


if __name__ == "__main__":
    main()
