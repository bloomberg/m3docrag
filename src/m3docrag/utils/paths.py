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
from dotenv import load_dotenv
load_dotenv() 

LOCAL_DATA_DIR = os.getenv("LOCAL_DATA_DIR", "/job/datasets")
LOCAL_EMBEDDINGS_DIR = os.getenv("LOCAL_EMBEDDINGS_DIR", "/job/embeddings")
LOCAL_MODEL_DIR = os.getenv("LOCAL_MODEL_DIR", "/job/model")
LOCAL_OUTPUT_DIR = os.getenv("LOCAL_OUTPUT_DIR", "/job/output")