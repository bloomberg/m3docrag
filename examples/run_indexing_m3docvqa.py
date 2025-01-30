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

from pathlib import Path

import faiss
import numpy as np
import torch
from loguru import logger
from tqdm.auto import tqdm

from m3docrag.datasets.m3_docvqa.dataset import M3DocVQADataset
from m3docrag.utils.args import parse_args


def main():
    args = parse_args()

    logger.info("Loading M3DocVQA")

    dataset = M3DocVQADataset(args)

    logger.info(f"Loading M3DocVQA -- all {args.retrieval_model_type} embeddings")

    if args.retrieval_model_type == "colpali":
        docid2embs = dataset.load_all_embeddings()
    elif args.retrieval_model_type == "colbert":
        docid2embs, docid2lens = dataset.load_all_embeddings()

    # len(docid2embs)
    # docid2embs_page_reduced = reduce_embeddings(docid2embs, dim='page')
    # docid2embs_token_reduced = reduce_embeddings(docid2embs, dim='token')
    # docid2embs_page_token_reduced = reduce_embeddings(docid2embs, dim='page_token')

    # flat_doc_embs = []
    # for doc_id, doc_emb in docid2embs.items():
    #     flat_doc_embs += [doc_emb]

    # flat_doc_embs  = torch.cat(flat_doc_embs, dim=0)

    # logger.info(flat_doc_embs.shape)

    d = 128
    quantizer = faiss.IndexFlatIP(d)

    if args.faiss_index_type == "flatip":
        index = quantizer

    elif args.faiss_index_type == "ivfflat":
        ncentroids = 1024
        index = faiss.IndexIVFFlat(quantizer, d, ncentroids)
    else:
        nlist = 100
        m = 8
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)

    logger.info("Flattening all PDF pages")

    all_token_embeddings = []
    token2pageuid = []

    if args.retrieval_model_type == "colpali":
        for doc_id, doc_emb in tqdm(docid2embs.items(), total=len(docid2embs)):
            # e.g., doc_emb - torch.Size([9, 1030, 128])
            for page_id in range(len(doc_emb)):
                page_emb = doc_emb[page_id].view(-1, d)
                all_token_embeddings.append(page_emb)

                page_uid = f"{doc_id}_page{page_id}"
                token2pageuid.extend([page_uid] * page_emb.shape[0])

    elif args.retrieval_model_type == "colbert":
        for doc_id, doc_emb in tqdm(docid2embs.items(), total=len(docid2embs)):
            doc_lens = docid2lens[doc_id]

            # e.g., doc_emb -  torch.Size([2089, 128])
            # e.g., doc_lens - tensor([258, 240, 251, 231, 229, 268, 235, 211, 166])

            all_token_embeddings.append(doc_emb)

            for page_id, page_len in enumerate(doc_lens):
                page_uid = f"{doc_id}_page{page_id}"
                token2pageuid.extend([page_uid] * page_len.item())

    logger.info(len(all_token_embeddings))

    all_token_embeddings = torch.cat(all_token_embeddings, dim=0)
    all_token_embeddings = all_token_embeddings.float().numpy()
    logger.info(all_token_embeddings.shape)
    logger.info(len(token2pageuid))

    logger.info("Creating index")

    index.train(all_token_embeddings)
    index.add(all_token_embeddings)

    Path(args.output_dir).mkdir(exist_ok=True)
    index_output_path = str(Path(args.output_dir) / "index.bin")
    logger.info(f"Saving index at {index_output_path}")

    faiss.write_index(index, index_output_path)

    logger.info("Running an example query")

    # Example query (should be np.float32)
    example_text_query_emb = np.random.randn(20, 128).astype(np.float32)

    # NN search
    k = 10
    D, I = index.search(example_text_query_emb, k)  # noqa E741

    # Sum the MaxSim scores across all query tokens for each document
    final_page2scores = {}

    # Iterate over query tokens
    for q_idx, query_emb in enumerate(example_text_query_emb):
        # Initialize a dictionary to hold document relevance scores
        curent_q_page2scores = {}

        for nn_idx in range(k):
            found_nearest_doc_token_idx = I[q_idx, nn_idx]

            page_uid = token2pageuid[
                found_nearest_doc_token_idx
            ]  # Get the document ID for this token

            # reconstruct the original score
            doc_token_emb = all_token_embeddings[found_nearest_doc_token_idx]
            score = (query_emb * doc_token_emb).sum()

            # MaxSim: aggregate the highest similarity score for each query token per document
            if page_uid not in curent_q_page2scores:
                curent_q_page2scores[page_uid] = score
            else:
                curent_q_page2scores[page_uid] = max(
                    curent_q_page2scores[page_uid], score
                )

        for page_uid, score in curent_q_page2scores.items():
            if page_uid in final_page2scores:
                final_page2scores[page_uid] += score
            else:
                final_page2scores[page_uid] = score

    # Sort documents by their final relevance score
    sorted_pages = sorted(final_page2scores.items(), key=lambda x: x[1], reverse=True)

    # Get the top-k document candidates
    top_k_pages = sorted_pages[:k]

    # Output the top-k document IDs and their scores
    logger.info("Top-k page candidates with scores:")
    for page_uid, score in top_k_pages:
        logger.info(f"{page_uid} with score {score}")


if __name__ == "__main__":
    main()
