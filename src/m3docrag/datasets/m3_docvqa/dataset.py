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

from datasets import load_dataset
from pathlib import Path
import torch
import safetensors
import json
import jsonlines
from copy import deepcopy
from tqdm.auto import tqdm
import PIL
from typing import List
from loguru import logger

from m3docrag.utils.paths import LOCAL_DATA_DIR, LOCAL_EMBEDDINGS_DIR
from m3docrag.utils.pdfs import get_images_from_pdf

class M3DocVQADataset(torch.utils.data.Dataset):
    def __init__(self, args):
        
        self.args = args
        
        # e.g., /job/datasets/m3-docvqa
        local_data_dir = Path(LOCAL_DATA_DIR) / args.data_name

        pdf_dir = local_data_dir / "splits" / f'pdfs_{args.split}'
        assert pdf_dir.exists(), pdf_dir
        self.pdf_dir = pdf_dir

        multimodalqa_data_dir = local_data_dir / "multimodalqa"

        mmqa_data_path = multimodalqa_data_dir / f"MMQA_{args.split}.jsonl"
        self.mmqa_data_path = mmqa_data_path
        
        data = []
        with jsonlines.open(mmqa_data_path) as reader:
            for i, obj in enumerate(reader):
                data.append(obj)
        logger.info(f"# Data {len(data)}")
        self.data = data

        split_supporting_doc_ids_path = local_data_dir / f"{args.split}_doc_ids.json"
        with open(split_supporting_doc_ids_path, 'r') as f:
            all_supporting_doc_ids = json.load(open(split_supporting_doc_ids_path))
            # dev: 3366
            # train: 24162
            logger.info(f"# supporting doc ids in split {args.split}: {len(all_supporting_doc_ids)}")
        self.all_supporting_doc_ids = all_supporting_doc_ids

    def __len__(self):
        if self.args.loop_unique_doc_ids:
            return len(self.all_supporting_doc_ids)

        if self.args.data_len is not None:
            return self.args.data_len

        return len(self.data)

    def load_all_embeddings(self):
        """Load all doc embeddings in memory"""

        emb_dir = Path(LOCAL_EMBEDDINGS_DIR) / self.args.embedding_name

        logger.info(f"Loading all doc embeddings from {emb_dir}")

        docid2embs = {}
        docid2lens = {}

        for idx in tqdm(range(len(self.all_supporting_doc_ids))):
        
            doc_id = self.all_supporting_doc_ids[idx]
            emb_path = Path(LOCAL_EMBEDDINGS_DIR) / self.args.embedding_name / f"{doc_id}.safetensors"
            assert emb_path.exists(), emb_path

            if self.args.retrieval_model_type == 'colpali':

                with safetensors.safe_open(emb_path, framework="pt", device='cpu') as f:

                    # [n_pages, n_tokens, dim]
                    doc_embs = f.get_tensor('embeddings')

            docid2embs[doc_id] = doc_embs.bfloat16()

        if self.args.retrieval_model_type == 'colpali':
            return docid2embs
        elif self.args.retrieval_model_type == 'colbert':
            return docid2embs, docid2lens


    def get_images_from_doc_id(self, doc_id: str) -> List[PIL.Image.Image]:
        pdf_path = self.pdf_dir / f"{doc_id}.pdf"
        page_images = get_images_from_pdf(pdf_path)
        return page_images
    

    def __getitem__(self, idx):
        if self.args.loop_unique_doc_ids:
            doc_id = self.all_supporting_doc_ids[idx]

            datum = {
                'doc_id': doc_id,
            }
            
            if self.args.retrieval_model_type == 'colpali':
                page_images = self.get_images_from_doc_id(doc_id)
                datum['images'] = page_images

            return datum

        # keys(['qid', 'question', 'answers', 'metadata', 'supporting_context'])
        datum = deepcopy(self.data[idx])

        supporting_doc_ids = []
        for obj in datum['supporting_context']:

            supporting_doc_ids.append(obj['doc_id'])
        datum['supporting_doc_ids'] = supporting_doc_ids

        return datum



if __name__ == '__main__':
    from m3docrag.utils.args import _example_args, parse_args

    args = parse_args(_example_args)

    dataset = M3DocVQADataset(
        args=args
    )