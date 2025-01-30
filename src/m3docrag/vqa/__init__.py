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



from typing import Union, List
from pathlib import Path
import torch

from m3docrag.vqa import internvl2
from m3docrag.vqa import idefics2
from m3docrag.vqa import idefics3
from m3docrag.vqa import florence2
from m3docrag.vqa import qwen2

ALL_VQA_MODEL_TYPES = ['florence2', 'idefics2', 'internvl2', 'idefics3', 'qwen2']

def init(
    model_name_or_path: Union[str, Path],
    model_type: str,
    **kwargs
):
    
    if 'internvl2' == model_type.lower():
        return internvl2.init(
            model_name_or_path=model_name_or_path,
            **kwargs
        )
    elif 'idefics2' == model_type.lower():
        return idefics2.init(
            model_name_or_path=model_name_or_path,
            **kwargs
        )
    elif 'idefics3' == model_type.lower():
        return idefics3.init(
            model_name_or_path=model_name_or_path,
            **kwargs
        )
    elif 'florence2' == model_type.lower():
        return florence2.init(
            model_name_or_path=model_name_or_path,
            **kwargs
        )
    elif 'qwen2' == model_type.lower():
        return qwen2.init(
            model_name_or_path=model_name_or_path,
            **kwargs
        )
    else:
        raise NotImplementedError(f"{model_type} is unsupported. Supported: {ALL_VQA_MODEL_TYPES}")
    

def generate(
    model_type: str,
    model,
    processor,
    **kwargs
) -> List[str]:

    if 'internvl2' == model_type.lower():
        return internvl2.generate(
            model=model,
            processor=processor,
            **kwargs
        )
    elif 'idefics2' == model_type.lower():
        return idefics2.generate(
            model=model,
            processor=processor,
            **kwargs
        )
    elif 'idefics3' == model_type.lower():
        return idefics3.generate(
            model=model,
            processor=processor,
            **kwargs
        )
    elif 'florence2' == model_type.lower():
        return florence2.generate(
            model=model,
            processor=processor,
            **kwargs
        )
    elif 'qwen2' == model_type.lower():
        return qwen2.generate(
            model=model,
            processor=processor,
            **kwargs
        )
    else:
        raise NotImplementedError(f"{model_type} is unsupported. Supported: {ALL_VQA_MODEL_TYPES}")
    

class VQAModel:

    def __init__(self, model_name_or_path: Union[str, Path], model_type: str, **kwargs):

        model_loaded = init(model_name_or_path=model_name_or_path, model_type=model_type, **kwargs)
        model = model_loaded['model']
        if 'tokenizer' in model_loaded:
            processor = model_loaded['tokenizer']
        else:
            processor = model_loaded['processor']

        if isinstance(model, torch.nn.Module):
            model = model.eval()

            # greedy decoding
            if hasattr(model, 'generation_config'):
                model.generation_config.temperature=None
                model.generation_config.top_p=None
                model.generation_config.top_k=None

        self.model = model
        self.processor = processor
        self.model_type = model_type

    def generate(self, images, question) -> str:
        responses =  generate(
            model_type=self.model_type,
            model=self.model,
            processor=self.processor,
            images=images,
            question=question,
        )
        assert isinstance(responses, list), responses

        out_text = responses[0]
        out_text = out_text.strip()

        return out_text
