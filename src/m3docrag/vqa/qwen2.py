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

from PIL import Image
import requests
import torch
from torchvision import io
from typing import Dict, List
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info


def init(
    model_name_or_path,
    dtype=torch.bfloat16,
    bits=16,
    attn_implementation="flash_attention_2",
    **kwargs,
):
    if bits == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype
        )
    else:
        bnb_config = None
    # Load the model in half-precision on the available device(s)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
        quantization_config=bnb_config,
        vision_config={"torch_dtype": dtype}
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name_or_path)

    return {
        'model': model,
        'processor': processor
    }

def generate(
    model,
    processor,
    question,
    images
) -> List[str]:

    image_content = [{"type": "image", "image": "dummy_content"}] * len(images)

    messages = [
        {
            "role": "user",
            "content": image_content + [{"type": "text", "text": question}]
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=images,
        # videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    p = next(iter(model.parameters()))

    inputs = inputs.to(p.device)

    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    assert isinstance(output_text, list), output_text

    return output_text