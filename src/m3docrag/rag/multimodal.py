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



from .base import RAGModelBase

import torch

from m3docrag.vqa import VQAModel
from m3docrag.retrieval import ColPaliRetrievalModel


class MultimodalRAGModel(RAGModelBase):
    def __init__(
        self,
        retrieval_model: ColPaliRetrievalModel,
        vqa_model: VQAModel = None
    ):
        self.retrieval_model = retrieval_model
        self.vqa_model = vqa_model

        self.retrieval_model.model.eval()

        if self.vqa_model is not None and isinstance(self.vqa_model.model, torch.nn.Module):
            self.vqa_model.model.eval()
    
    
    def run_vqa(
        self,
        images,
        question,
    ) -> str:

        response = self.vqa_model.generate(images=images, question=question)
        assert isinstance(response, str), type(response)

        return response