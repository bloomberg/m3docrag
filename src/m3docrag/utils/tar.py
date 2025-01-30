# Copyright 2024 Bloomberg Finance L.P.
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
#
# SPDX-License-Identifier: Apache-2.0

import os
import tarfile
from loguru import logger

def make_tarfile(source_dir, output_filename):
    logger.info(f"Compressing {source_dir} to {output_filename} ...")
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname='.')
    logger.info(f"Compression done!")


def extract_tarfile(input_filename, target_dir):
    logger.info(f"Extracting {input_filename} to {target_dir} ...")
    with tarfile.open(input_filename) as f:
        f.extractall(target_dir)
    logger.info(f"Extraction done!")