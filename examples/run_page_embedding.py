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

import accelerate
import safetensors
import torch
import transformers
from accelerate import Accelerator
from loguru import logger
from tqdm import tqdm

from m3docrag.datasets.m3_docvqa import M3DocVQADataset
from m3docrag.retrieval import ColPaliRetrievalModel
from m3docrag.utils.args import parse_args
from m3docrag.utils.distributed import (
    barrier,
    global_rank,
    is_distributed,
    local_rank,
    log_runtime_info,
    print_gpu_stats,
)
from m3docrag.utils.paths import (
    LOCAL_DATA_DIR,
    LOCAL_MODEL_DIR,
)

logger.info(torch.__version__)
logger.info(transformers.__version__)
logger.info(accelerate.__version__)


def main():
    args = parse_args()

    log_runtime_info()
    print_gpu_stats()

    accelerator = Accelerator()

    if not is_distributed() or global_rank() == 0:
        logger.info(f"Process {global_rank()}:{local_rank()} - args {args}")

    if is_distributed():
        barrier()

    local_data_dir = Path(LOCAL_DATA_DIR) / args.data_name
    local_retrieval_model_dir = (
        Path(LOCAL_MODEL_DIR) / args.retrieval_model_name_or_path
    )
    local_retrieval_adapter_model_dir = (
        Path(LOCAL_MODEL_DIR) / args.retrieval_adapter_model_name_or_path
    )

    # Download datasets / model checkpoints
    if not is_distributed() or global_rank() == 0:
        if not local_data_dir.exists():
            raise ValueError(f"Data directory {local_data_dir} does not exist")

        assert args.use_retrieval, args.use_retrieval

        if not local_retrieval_model_dir.exists():
            raise ValueError(
                f"Retrieval model directory {local_retrieval_model_dir} does not exist"
            )

        if args.retrieval_model_type == "colpali":
            if not local_retrieval_adapter_model_dir.exists():
                raise ValueError(
                    f"Retrieval adapter model directory {local_retrieval_adapter_model_dir} does not exist"
                )

    if is_distributed():
        barrier()

    if args.retrieval_model_type == "colpali":
        colpali_model = ColPaliRetrievalModel(
            backbone_name_or_path=local_retrieval_model_dir,
            adapter_name_or_path=local_retrieval_adapter_model_dir,
        )
        retrieval_model = colpali_model

    if args.data_name == "m3-docvqa":
        dataset = M3DocVQADataset(args=args)

        def collate_fn(examples):
            out = {}
            if args.retrieval_model_type == "colpali":
                for k in ["doc_id", "images"]:
                    out[k] = [ex[k] for ex in examples]
            return out

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=1,
        shuffle=False,
        batch_sampler=None,
        sampler=None,
        drop_last=False,
        num_workers=args.dataloader_num_workers,
    )

    retrieval_model.model, data_loader = accelerator.prepare(
        retrieval_model.model, data_loader
    )

    all_results = []

    save_dir = Path(args.output_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Results will be saved at: {save_dir}")

    for i, datum in enumerate(tqdm(data_loader)):
        print(f"{i} / {len(data_loader)}")

        if args.data_name == "mp-docvqa":
            page_name = datum["page_name"][0]
            logger.info(page_name)
        else:
            doc_id = datum["doc_id"][0]
            logger.info(doc_id)

        if args.retrieval_model_type == "colpali":
            images = datum["images"][0]

            doc_embs = colpali_model.encode_images(
                images=images,
                batch_size=args.per_device_eval_batch_size,
                to_cpu=True,
                use_tqdm=False,
            )

            # [n_pages, n_tokens, emb_dim]
            doc_embs = torch.stack(doc_embs, dim=0)

        # Store embedding as BF16 by default
        doc_embs = doc_embs.to(torch.bfloat16)

        logger.info(doc_embs.shape)
        if args.retrieval_model_type == "colpali":
            logger.info(doc_embs[0, 0, :5])

        # Save the embedding
        if args.data_name == "mp-docvqa":
            local_save_fname = f"{page_name}.safetensors"
        else:
            local_save_fname = f"{doc_id}.safetensors"
        local_save_path = save_dir / local_save_fname

        if args.retrieval_model_type == "colpali":
            safetensors.torch.save_file({"embeddings": doc_embs}, local_save_path)

        all_results.append({"save_path": local_save_path})

    logger.info(
        f"Process {global_rank()}:{local_rank()} Results correctly saved at {save_dir}"
    )

    if is_distributed():
        barrier()


if __name__ == "__main__":
    main()
