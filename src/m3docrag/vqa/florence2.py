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

import torch
from transformers import AutoProcessor, AutoModelForCausalLM 
from PIL import Image

from typing import Union, List


def init(
    model_name_or_path,
    dtype=torch.bfloat16,
    # model_max_length=None,
    **kwargs,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        # model_max_length=model_max_length
        )
    
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
    
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        module = model.module
    else:
        module = model
    answer = generate_caption(module, processor=processor, images=images, text_input=question)
    return answer



@torch.no_grad()
def generate_caption(
    model,
    processor,
    images=None,
    #  task_prompt="<MORE_DETAILED_CAPTION>",
    text_input=None,
    input_ids=None,
    pixel_values=None,
    max_new_tokens=77,
    num_beams=1,
    do_sample=False,
    decode_text=True,
    **generate_kwargs
):

    if input_ids is None and pixel_values is None:
        if isinstance(images, Image.Image):
            images = [images]

        B = len(images)

        if isinstance(text_input, str):
            text_input = [text_input] * B

        inputs = processor(
            text=text_input,
            images=images,
            return_tensors="pt")

        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"]

    p = next(iter(model.parameters()))
    device = p.device
    dtype = p.dtype
    
    generated_ids = model.generate(
      input_ids=input_ids.to(device),
      pixel_values=pixel_values.to(device, dtype),
      max_new_tokens=max_new_tokens,
      num_beams=num_beams,
      do_sample=do_sample,
      **generate_kwargs
    )
    if decode_text:
        out_text = decode_predictions(processor, generated_ids)

        return out_text
    else:
        return generated_ids
    
def decode_predictions(processor, generated_ids):
    B = len(generated_ids)

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    out_text = []
    for i in range(B):
        out_text.append(generated_text[i].replace('<s>', '').replace('</s>', '').replace('<pad>', '').strip())
    return out_text

# def load_model_from_ckpt(
#         ckpt_dir,
#         load_abstractor=False,
#         num_input_tokens=576,
#         num_query_tokens=64,
#         proj_type='c-abs',
#         projection_dim=1024,
#         strict=False
#     ):
#     florence_config = Florence2Config.from_pretrained(ckpt_dir)

#     if load_abstractor:
#         abstractor_config = AbstractorConfig(
#             num_input_tokens=num_input_tokens,
#             num_query_tokens=num_query_tokens,
#             proj_type=proj_type,
#             projection_dim=projection_dim,
#         )
#         florence_config.abstractor_config = abstractor_config

#     florence_config.vision_config.model_type = 'davit'
    
#     model = Florence2ForConditionalGeneration(config=florence_config)

    
#     ckpt_path = Path(ckpt_dir) / 'model.safetensors'
#     if ckpt_path.exists():
#         logger.info(f"loading checkpoints from {ckpt_path}")
#         state_dict = safetensors.torch.load_file(ckpt_path, device="cpu")

#     else:
#         ckpt_path = Path(ckpt_dir) / 'pytorch_model.bin'
#         logger.info(f"loading checkpoints from {ckpt_path}")
#         state_dict = torch.load(
#             ckpt_path,
#             map_location="cpu",
#         )

#     load_result = model.load_state_dict(state_dict, strict=strict)
#     logger.info(load_result)

#     return model

