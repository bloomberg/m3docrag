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

"""Downloads the portion of the multimodalqa dataset from https://github.com/allenai/multimodalqa/tree/master/dataset 
that is useful for creating the m3docvqa dataset.
"""
import gzip
import requests
from loguru import logger
from pathlib import Path


def download_file(url: str, output_path: str) -> None:
    """Downloads a file from a given URL and saves it to the specified output path.

    Args:
        url (str): The URL of the file to download.
        output_path (str): The path where the downloaded file will be saved.

    Raises:
        requests.exceptions.RequestException: If the file could not be downloaded.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"File downloaded successfully: {output_path}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download file from {url}: {e}")
        raise

def decompress_gz_file(input_path: str | Path, output_path: str | Path) -> None:
    """
    Decompresses a `.gz` file into its original format.

    Args:
        input_path (str | Path): Path to the `.gz` file.
        output_path (str | Path): Path where the decompressed file will be written.

    Raises:
        ValueError: If the input path does not exist or is not a file.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.is_file():
        raise ValueError(f"The input file {input_path} does not exist or is not a file.")

    with gzip.open(input_path, "rb") as f_in, open(output_path, "wb") as f_out:
        f_out.write(f_in.read())
    logger.info(f"Decompressed {input_path} to {output_path}")

def download_and_decompress_mmqa(output_directory: str | Path) -> None:
    """
    Downloads and decompresses the MultiModalQA dataset files into the specified directory.

    Args:
        output_directory (str | Path): The directory where the files will be stored.

    Steps:
        1. Creates the output directory if it doesn't exist.
        2. Downloads the `.jsonl.gz` files.
        3. Decompresses each `.gz` file into its `.jsonl` format.
        4. Removes the `.gz` files after decompression.

    Raises:
        requests.exceptions.RequestException: If any of the files could not be downloaded.
    """
    # Define base URL and file names
    base_url = "https://github.com/allenai/multimodalqa/raw/refs/heads/master/dataset/"
    files = [
        "MMQA_texts.jsonl.gz", 
        "MMQA_tables.jsonl.gz", 
        "MMQA_images.jsonl.gz",
        "MMQA_dev.jsonl.gz",
        "MMQA_train.jsonl.gz",
        ]
        
    output_directory = Path(output_directory)

    # Ensure the output directory exists
    if not output_directory.exists():
        output_directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {output_directory}")

    for file_name in files:
        compressed_path = output_directory / file_name
        decompressed_path = output_directory / file_name.replace(".gz", "")

        try:
            # Step 1: Download the file
            logger.info(f"Downloading {file_name}...")
            download_file(base_url + file_name, compressed_path)

            # Step 2: Decompress the file
            logger.info(f"Decompressing {file_name}...")
            decompress_gz_file(compressed_path, decompressed_path)

            # Step 3: Remove the compressed file
            compressed_path.unlink()
            logger.info(f"Removed compressed file: {compressed_path}")
        except Exception as e:
            logger.error(f"Error processing {file_name}: {e}")
            raise
