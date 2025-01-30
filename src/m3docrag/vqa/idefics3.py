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

from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers import BitsAndBytesConfig

from typing import Union, List

def init(
    model_name_or_path,
    bits=4,
    dtype=torch.bfloat16,
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

    model = AutoModelForVision2Seq.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True                         
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
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

    messages = idefics3_create_message(images=images, question=question)

    examples = [{'images': images, 'messages': messages}]
    batch = idefics3_collate_fn(examples, processor)

    for k in batch:
        batch[k] = batch[k].to(module.device)

    generated_ids = module.generate(**batch, max_new_tokens=50, do_sample=False)
    answer = processor.batch_decode(
        generated_ids[:, batch.input_ids.size(1):],
        skip_special_tokens=True)
    return answer
    

def idefics3_collate_fn(examples,
                       processor,
                       is_train=False,
                       image_token_id=None,
                       ):

    texts = []
    images = []
    for example in examples:
        prompt = processor.apply_chat_template(example['messages'], add_generation_prompt=not is_train)
        texts.append(prompt)
        images.append(example['images'])

    if is_train:
        batch = processor(text=texts, images=images, return_tensors="pt", 
                        padding=True, truncation=True,
                        # max_length=model_max_length
                        )
        
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        labels[labels == image_token_id] = -100
        batch["labels"] = labels
    else:
        batch = processor(text=texts, images=images, return_tensors="pt",
                            padding=True,
                            truncation=True,
                            # max_length=model_max_length
                            )
        
    return batch
        


def idefics3_create_message(images, question, is_train=False, target_text=None):
    content = []
    for page_i in range(len(images)):
        # content += [{"type": "text", "text": f"page {page_i}: "}]
        content += [{"type": "image"}]
    content += [{"type": "text", "text": question}]
    messages = [{"role": "user", "content": content}]

    if is_train:
        messages += [{"role": "assistant", "content": [{"type": "text", "text": target_text}]}]
    
    return messages
