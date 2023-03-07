#!/usr/bin/env python3
# This source codes based from Feiteng Li, customized by atlonxp
# Copyright    2023                           (authors: Feiteng Li)
# Copyright    2023                           (authors: Dr Watthanasak Jeamwatthanachai
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file displays duration statistics of utterances in the manifests.
You can use the displayed value to choose minimum/maximum duration
to remove short and long utterances during the training.
"""


from lhotse import load_manifest_lazy


def main():
    for part in ["train", "dev", "test"]:
        print(f"##  {part}")
        cuts = load_manifest_lazy(f"./data/tokenized/cuts_{part}.jsonl.gz")
        cuts.describe()
        print("\n")


if __name__ == "__main__":
    main()
