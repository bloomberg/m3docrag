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
from torch.utils.data import DataLoader
from tqdm import tqdm
# from transformers import AutoProcessor
from PIL import Image
from typing import List

from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.models import ColQwen2, ColQwen2Processor

def init(
    backbone_name_or_path="/job/model/colpaligemma-3b-pt-448-base",
    adapter_name_or_path= "/job/model/colpali-v1.2",
    dtype=torch.bfloat16,
):
    """
    Load ColPali Model and Processor from (locally downloaded) HF checkpoint.
    
    Args:
        - backbone_model_name_or_path: downloaded from https://huggingface.co/vidore/colpaligemma-3b-pt-448-base
        - adapter_name_or_path: downloaded from https://huggingface.co/vidore/colpali-v1.2
    Return:
        - model
        - processor
    """

    kwargs = {}
    model_class = ColPali
    processor_class = ColPaliProcessor
    if 'colqwen' in str(adapter_name_or_path):
        model_class = ColQwen2
        processor_class = ColQwen2Processor
        kwargs['attn_implementation'] = "flash_attention_2"

    model = model_class.from_pretrained(
        backbone_name_or_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        **kwargs
    ).eval()

    model.load_adapter(adapter_name_or_path)
    processor = processor_class.from_pretrained(adapter_name_or_path)

    return model, processor


def encode_images(
    model,
    processor,
    images: List[Image.Image],
    batch_size: int = 4,
    to_cpu: bool = False,
    use_tqdm: bool = False,
    collate_fn=None,
    return_doclens: bool = False
    ):
    """Create document embeddings with ColPali

    Args:
        model
        processor
        images (List[Image.Image])
            (n_pages)
        batch_size (int, optional):
            batch size. Defaults to 4.
        to_cpu (bool, optional):
            whether to save embeddings in cpu tensors. Defaults to False.
        use_tqdm (bool, optional):
            whether to show tqdm progress bar. Defaults to False.
        collate_fn (_type_, optional):
            custom collate_fn for document dataloader. Defaults to None.
        return_doclens (bool, optional):
            whether to output the number of pages. Defaults to False.

    Returns:
        doc_embs: List[torch.tensor]
            visual embedding of documents (n_pages, n_tokens, n_dimension)
        (optional) doclens
            number of pages
    """    

    if collate_fn is None:
        collate_fn = processor.process_images

    # run inference - docs
    dataloader = DataLoader(
        images,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    doc_embs = []
    if return_doclens:
        doclens = []
    if use_tqdm:
        dataloader = tqdm(dataloader)
    for batch_doc in dataloader:
        with torch.no_grad():
            batch_doc = {k: v.to(model.device) for k, v in batch_doc.items()}
            embeddings_doc = model(**batch_doc)
            if to_cpu:
                embeddings_doc = embeddings_doc.to("cpu")
            if return_doclens:
                _doclens = batch_doc.attention_mask.squeeze(-1).sum(-1).tolist()
                doclens.extend(_doclens)
        doc_embs.extend(list(torch.unbind(embeddings_doc)))

    if return_doclens:
        doc_embs, doclens
    else:
        return doc_embs


def encode_queries(
    model,
    processor,
    queries: List[str],
    batch_size: int = 4,
    to_cpu: bool = False,
    use_tqdm: bool = False,
    collate_fn=None,
    ):
    """Create query embeddings with ColPali

    Args:
        model
        processor
        queries (List[str]):
            text queries (n_queries,)
        batch_size (int, optional):
            batch size. Defaults to 4.
        to_cpu (bool, optional):
            whether to save embeddings in cpu tensors. Defaults to False.
        use_tqdm (bool, optional):
            whether to show tqdm progress bar. Defaults to False.
        collate_fn (_type_, optional):
            custom collate_fn for document dataloader. Defaults to None.
    Returns:
        query_embs: List[torch.tensor]
            embedding of queries (n_queries, n_tokens, n_dimension)
    """    

    if collate_fn is None:
        collate_fn = processor.process_queries

    # run inference - queries
    dataloader = DataLoader(
        queries,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    query_embs = []
    if use_tqdm:
        dataloader = tqdm(dataloader)
    for batch_query in dataloader:
        with torch.no_grad():
            batch_query = {k: v.to(model.device) for k, v in batch_query.items()}
            embeddings_query = model(**batch_query)
            if to_cpu:
                embeddings_query = embeddings_query.to("cpu")
        query_embs.extend(list(torch.unbind(embeddings_query)))
    return query_embs


def retrieve(
    model,
    processor,
    docs=None,
    query=None,
    doc_embeds=None,
    query_embeds=None,
    to_cpu=False,
    batch_size=1,
    use_tqdm=False,
    return_top_1=True
):
    """Find the right document image with colpali
    """
    if doc_embeds is None:
        doc_embeds = encode_images(
            model, processor,
            images=docs,
            batch_size=batch_size,
            use_tqdm=use_tqdm,
            to_cpu=to_cpu,
        )

    if query_embeds is None:
        query_embeds = encode_queries(
            model, processor,
            queries=[query],
            batch_size=1,
            use_tqdm=use_tqdm,
            to_cpu=to_cpu,
        )

    qs = query_embeds
    ds = doc_embeds
    
    qs = [q.to(ds[0].dtype) for q in qs]

    scores = processor.score_multi_vector(qs, ds)

    if return_top_1:
        return scores.argmax(axis=1)
    else:
        return scores


class ColPaliRetrievalModel:

    def __init__(self, 
        backbone_name_or_path="/job/model/colpaligemma-3b-pt-448-base",
        adapter_name_or_path= "/job/model/colpali-v1.2",
        dtype=torch.bfloat16,
    ):
        model, processor = init(backbone_name_or_path=backbone_name_or_path,
                                adapter_name_or_path=adapter_name_or_path,
                                dtype=dtype,
        )
        self.model = model.eval()
        self.processor = processor

    def encode_queries(self,
        queries: List[str],
        batch_size: int = 4,
        to_cpu: bool = False,
        use_tqdm: bool = False,
        collate_fn=None
    ):
        return encode_queries(
            model=self.model,
            processor=self.processor,
            queries=queries,
            batch_size=batch_size,
            to_cpu=to_cpu,
            use_tqdm=use_tqdm,
            collate_fn=collate_fn)

    
    def encode_images(self,
        images: List[Image.Image],
        batch_size: int = 4,
        to_cpu: bool = False,
        use_tqdm: bool = False,
        collate_fn=None,
        return_doclens: bool = False
    ):
        return encode_images(
            model=self.model,
            processor=self.processor,
            images=images,
            batch_size=batch_size,
            to_cpu=to_cpu,
            use_tqdm=use_tqdm,
            collate_fn=collate_fn,
            return_doclens=return_doclens,
        )

    def retrieve(self,
        docs=None,
        query=None,
        doc_embeds=None,
        doc_lens=None,
        query_embeds=None,
        to_cpu=False,
        batch_size=1,
        use_tqdm=False,
        return_top_1=True
    ):
        
        return retrieve(
            model=self.model,
            processor=self.processor,
            docs=docs,
            query=query,
            doc_embeds=doc_embeds,
            query_embeds=query_embeds,
            to_cpu=to_cpu,
            batch_size=batch_size,
            use_tqdm=use_tqdm,
            return_top_1=return_top_1
        )