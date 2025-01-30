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


from tqdm.auto import tqdm


def reduce_embeddings(docid2embs, dim='page', show_progress=True):
    """Summarize document embedding by reducing (averaging) specific dimensions
    
    Input embedding:
        [n_pages, n_tokens, emb_dim]

    Output embedding:

    - reduction_dim == 'page'
        [1, n_tokens, emb_dim]

    - reduction_dim == 'token'
        [1, n_pages, emb_dim]

    - reduction_dim == 'page_token'
        [1, 1, emb_dim]
    """

    assert dim in ['page', 'token', 'page_token'], f"{dim}"

    new_docid2embs = {}

    for doc_id in tqdm(
        list(docid2embs.keys()),
        disable=not show_progress
    ):
        # [n_pages, n_tokens, dim]
        embs = docid2embs[doc_id]

        emb_dim = embs.size(-1)

        if dim == 'page':
            # [n_tokens, dim]
            new_embs = embs.mean(dim=0)
        elif dim == 'token':
            # [n_pages, dim]
            new_embs = embs.mean(dim=1)
        elif dim == 'page_token':
            # [1, dim]
            new_embs = embs.mean(dim=0).mean(dim=0)

        new_docid2embs[doc_id] = new_embs.view(1, -1, emb_dim)
    
    return new_docid2embs


def get_top_k_pages(docid2scores: dict, k: int):
    """
    # Example usage:
    docid2scores = {
        "doc1": [10, 50, 30],
        "doc2": [40, 20, 60],
        "doc3": [70, 90]
    }

    k = 3
    top_k_pages = get_top_k_pages(docid2scores, k)
    print(top_k_pages)

    -> [('doc3', 1, 90), ('doc3', 0, 70), ('doc2', 2, 60)]
    """
    # Flatten the dictionary into a list of tuples (doc_id, page_index, score)
    flattened_scores = [
        (doc_id, page_index, score)
        for doc_id, scores in docid2scores.items()
        for page_index, score in enumerate(scores)
    ]

    # Sort by score in descending order
    flattened_scores.sort(key=lambda x: x[2], reverse=True)

    # Get the top-k entries
    top_k_pages = flattened_scores[:k]

    return top_k_pages


def get_top_k_pages_single_page_from_each_doc(docid2scores: dict, k: int):
    """
    # Example usage:
    docid2scores = {
        "doc1": [10, 50, 30],
        "doc2": [40, 20, 60],
        "doc3": [70, 90]
    }

    k = 2
    top_k_pages = get_top_k_pages_single_page_from_each_doc(docid2scores, k)
    print(top_k_pages)

    -> [('doc3', 1, 90), ('doc2', 2, 60)]
    """
    # First, get the highest scoring page for each document
    highest_per_doc = [
        (doc_id, max(enumerate(scores), key=lambda x: x[1]))  # (doc_id, (page_index, score))
        for doc_id, scores in docid2scores.items()
    ]

    # Flatten the structure to (doc_id, page_index, score)
    highest_per_doc_flat = [(doc_id, page_index, score) for doc_id, (page_index, score) in highest_per_doc]

    # Sort by score in descending order
    highest_per_doc_flat.sort(key=lambda x: x[2], reverse=True)

    # Get the top-k entries
    top_k_pages = highest_per_doc_flat[:k]

    return top_k_pages
