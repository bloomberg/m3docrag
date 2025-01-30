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

"""Wiki Mapper.

This module provides functionality to parse multimodalqa JSONL files that has been already downloaded that contains 'id' and 'url' mappings,
merge them into a single mapping, and save the result to a JSONL file.

Each JSONL file should contain one JSON object per line with the following structure:
{
    "title": "Article Title",
    "url": "https://en.wikipedia.org/wiki/Article_Title",
    "id": "unique_id",
    "text": "Text description of the article."
}
"""

import json
from pathlib import Path
from loguru import logger


def parse_jsonl(file_path: str | Path) -> dict[str, str]:
    """Parses a JSONL file from the multimodalqa dataset to extract a mapping of 'id' to 'url'.

    Args:
        file_path (str | Path): Path to the JSONL file.

    Returns:
        dict[str, str]: A dictionary mapping each 'id' to its corresponding 'url'.

    Raises:
        FileNotFoundError: If the JSONL file does not exist.
        ValueError: If the file contains invalid JSON lines.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    id_url_mapping = {}
    try:
        with file_path.open("r") as file:
            for line in file:
                data = json.loads(line.strip())
                entry_id = data.get("id")
                url = data.get("url")
                if entry_id and url:
                    id_url_mapping[entry_id] = url
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON in file {file_path}: {e}")
        raise ValueError(f"Invalid JSON in file {file_path}: {e}")

    logger.info(f"Parsed {len(id_url_mapping)} entries from {file_path}")
    return id_url_mapping


def merge_mappings(mappings: list[dict[str, str]]) -> dict[str, str]:
    """Merges multiple mappings into a single dictionary.

    Args:
        mappings (list[dict[str, str]]): A list of dictionaries containing 'id' to 'url' mappings.

    Returns:
        dict[str, str]: A merged dictionary containing all 'id' to 'url' mappings.
    """
    merged_mapping = {}
    for mapping in mappings:
        merged_mapping.update(mapping)
    logger.info(f"Merged {len(mappings)} mappings with a total of {len(merged_mapping)} entries.")
    return merged_mapping


def save_mapping_to_jsonl(mapping: dict[str, str], output_file: str | Path) -> None:
    """Saves the 'id'-to-'url' mapping to a JSONL file.

    Args:
        mapping (dict[str, str]): The dictionary containing 'id' to 'url' mappings.
        output_file (str | Path): The path to the output JSONL file.

    Raises:
        IOError: If the file cannot be written.
    """
    output_file = Path(output_file)
    try:
        with output_file.open("w") as file:
            for entry_id, url in mapping.items():
                json.dump({"id": entry_id, "url": url}, file)
                file.write("\n")
        logger.info(f"Mapping saved to {output_file}")
    except IOError as e:
        logger.error(f"Error writing to file {output_file}: {e}")
        raise


def generate_wiki_links_mapping(
    text_file: str | Path, image_file: str | Path, table_file: str | Path, output_file: str | Path = "id_url_mapping.jsonl"
) -> None:
    """Orchestrates the process of parsing input files, merging mappings, and saving the result to JSONL.

    Args:
        text_file (str | Path): Path to the JSONL file containing text data with 'id' and 'url' fields.
        image_file (str | Path): Path to the JSONL file containing image data with 'id' and 'url' fields.
        table_file (str | Path): Path to the JSONL file containing table data with 'id' and 'url' fields.
        output_file (str | Path): Path to save the output JSONL file. Defaults to 'id_url_mapping.jsonl'.

    Raises:
        Exception: If any part of the pipeline fails.
    """
    try:
        # Parse input files
        logger.info("Parsing JSONL files...")
        text_mapping = parse_jsonl(text_file)
        image_mapping = parse_jsonl(image_file)
        table_mapping = parse_jsonl(table_file)

        # Merge mappings
        logger.info("Merging mappings...")
        merged_mapping = merge_mappings([text_mapping, image_mapping, table_mapping])

        # Save the merged mapping
        logger.info("Saving merged mapping to output file...")
        save_mapping_to_jsonl(merged_mapping, output_file)
        logger.info(f"Mapping successfully generated and saved to {output_file}")
    except Exception as e:
        logger.error(f"Error generating wiki links mapping: {e}")
        raise
