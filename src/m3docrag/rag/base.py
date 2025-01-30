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


from typing import List, Dict, Tuple
from tqdm.auto import tqdm
import torch
import numpy as np

from .utils import get_top_k_pages, get_top_k_pages_single_page_from_each_doc

class RAGModelBase:
    def __init__(
        self,
        retrieval_model=None,
        qa_model=None,
        vqa_model=None,
    ):
        """Base class for RAG pipeline
        
        - retrieval_model: arbitrary retrieval model (e.g., ColPali / ColBERT)
        - qa_model: arbitrary text-only QA model (e.g., LLama3)
        - vqa_model: arbitrary VQA model (e.g., InternVL2, GPT4-o)
        
        """
        self.retrieval_model = retrieval_model
        self.qa_model = qa_model
        self.vqa_model = vqa_model

        if self.retrieval_model is not None:
            self.retrieval_model.model.eval()
        if self.vqa_model is not None:
            self.vqa_model.model.eval()
        if self.qa_model is not None:
            self.qa_model.model.eval()
    
    
    def retrieve_pages_from_docs(
        self,
        query: str,
        docid2embs: Dict[str, torch.Tensor],
        docid2lens: Dict[str, torch.Tensor] = None,
        
        index = None,
        token2pageuid = None,
        all_token_embeddings = None,

        n_return_pages: int = 1,
        single_page_from_each_doc: bool = False,
        show_progress=False,
    ) -> List[Tuple]:
        """
        Given text query and pre-extracted document embedding,
        calculate similarity scores and return top-n pages

        Args:
            - query (str): a text query to call retrieval model  
            - docid2embs (Dict[str, tensor]): collection of document embeddings
                key: document_id
                value: torch.tensor of size (n_tokens, emb_dim)
            - index: faiss index
            - n_return_pages (int): number of pages to return
            - single_page_from_each_doc (bool): if true, only single page is retrieved from each PDF document.

        Return:
            retrieval_results
            [(doc_id, page_idx, scores)...]
        """


        if index is not None:

            # [n_query_tokens, dim]
            query_emb = self.retrieval_model.encode_queries([query])[0]
            query_emb = query_emb.cpu().float().numpy().astype(np.float32)

            # NN search
            k = n_return_pages
            D, I = index.search(query_emb, k)

            # Sum the MaxSim scores across all query tokens for each document
            final_page2scores = {}

            # Iterate over query tokens
            for q_idx, query_emb in enumerate(query_emb):

                # Initialize a dictionary to hold document relevance scores
                curent_q_page2scores = {}

                for nn_idx in range(k):
                    found_nearest_doc_token_idx = I[q_idx, nn_idx]

                    page_uid = token2pageuid[found_nearest_doc_token_idx]  # Get the document ID for this token

                    # reconstruct the original score
                    doc_token_emb = all_token_embeddings[found_nearest_doc_token_idx]
                    score = (query_emb * doc_token_emb).sum()

                    # MaxSim: aggregate the highest similarity score for each query token per document
                    if page_uid not in curent_q_page2scores:
                        curent_q_page2scores[page_uid] = score
                    else:
                        curent_q_page2scores[page_uid] = max(curent_q_page2scores[page_uid], score)

            
                for page_uid, score in curent_q_page2scores.items():
                    if page_uid in final_page2scores:
                        final_page2scores[page_uid] += score
                    else:
                        final_page2scores[page_uid] = score
                    
            # Sort documents by their final relevance score
            sorted_pages = sorted(final_page2scores.items(), key=lambda x: x[1], reverse=True)

            # Get the top-k document candidates
            top_k_pages = sorted_pages[:k]

            
            # [(doc_id, page_idx, scores)...]

            sorted_results = []
            for page_uid, score in top_k_pages:
                # logger.info(f"{page_uid} with score {score}")

                # page_uid = f"{doc_id}_page{page_id}"
                doc_id = page_uid.split('_page')[0]
                page_idx = int(page_uid.split('_page')[-1])
                sorted_results.append((doc_id, page_idx, score.item()))

            return sorted_results

        docid2scores = {}
        for doc_id, doc_embs in tqdm(
            docid2embs.items(),
            total=len(docid2embs),
            disable = not show_progress,
            desc=f"Calculating similarity score over documents"
        ):
            doc_lens = None
            if docid2lens is not None:
                doc_lens = docid2lens[doc_id]

            scores = self.retrieval_model.retrieve(
                query=query,
                doc_embeds=doc_embs,
                doc_lens=doc_lens,
                to_cpu=True,
                return_top_1=False
            )
            scores = scores.flatten().tolist()
            docid2scores[doc_id] = scores

        # find the pages with top scores
        if single_page_from_each_doc:
            return get_top_k_pages_single_page_from_each_doc(docid2scores=docid2scores, k=n_return_pages)
        else:
            return get_top_k_pages(docid2scores=docid2scores, k=n_return_pages)

        
    def run_qa(self):
        raise NotImplementedError

    def run_vqa(self):
        raise NotImplementedError