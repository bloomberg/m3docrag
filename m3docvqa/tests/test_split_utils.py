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

import pytest
from pathlib import Path
import shutil
import json
import jsonlines
from unittest.mock import MagicMock, patch
from m3docvqa.split_utils import create_split_dirs


@pytest.fixture
def mock_pdf_directory(tmp_path):
    # Create a temporary directory for PDFs
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    # Add some mock PDF files
    (pdf_dir / "doc1.pdf").write_text("PDF content for doc1")
    (pdf_dir / "doc2.pdf").write_text("PDF content for doc2")
    return pdf_dir


@pytest.fixture
def mock_metadata_file(tmp_path):
    # Create a temporary metadata file in JSONL format
    metadata_file = tmp_path / "MMQA_train.jsonl"
    data = [
        {"supporting_context": [{"doc_id": "doc1"}]},
        {"supporting_context": [{"doc_id": "doc2"}]}
    ]
    with jsonlines.open(metadata_file, mode='w') as writer:
        writer.write_all(data)
    return metadata_file


@pytest.fixture
def mock_target_directory(tmp_path):
    return tmp_path / "target"


def test_create_split_dirs(mock_pdf_directory, mock_metadata_file, mock_target_directory):
    """Test the create_split_dirs function."""
    # Prepare the split directory
    split = "train"
    
    # Call the function to create split directories
    create_split_dirs(
        all_pdf_dir=mock_pdf_directory,
        target_dir_base=mock_target_directory,
        split_metadata_file=mock_metadata_file,
        split=split
    )
    
    # Assert that the target directory exists and contains the expected PDF files
    target_dir = mock_target_directory / f"pdfs_{split}"
    assert target_dir.exists(), f"Directory {target_dir} was not created"
    assert (target_dir / "doc1.pdf").exists(), "doc1.pdf was not copied"
    assert (target_dir / "doc2.pdf").exists(), "doc2.pdf was not copied"


def test_create_split_dirs_missing_pdf(mock_metadata_file, mock_target_directory):
    """Test create_split_dirs when PDF files are missing."""
    # Prepare the split directory
    split = "train"
    all_pdf_dir = Path("non_existing_pdf_dir")
    
    # Call the function and verify that the missing PDFs are handled correctly
    create_split_dirs(
        all_pdf_dir=all_pdf_dir,
        target_dir_base=mock_target_directory,
        split_metadata_file=mock_metadata_file,
        split=split
    )
    
    target_dir = mock_target_directory / f"pdfs_{split}"
    assert target_dir.exists(), f"Directory {target_dir} was not created"
    assert not (target_dir / "doc1.pdf").exists(), "doc1.pdf should not exist"
    assert not (target_dir / "doc2.pdf").exists(), "doc2.pdf should not exist"


@pytest.mark.parametrize("split, expected_error", [
    ("test_split", ValueError),  # Invalid split type
    (None, ValueError),  # Missing split
])
def test_create_split_dirs_invalid_split_type(mock_pdf_directory, mock_metadata_file, mock_target_directory, split, expected_error):
    """Test invalid split types in create_split_dirs."""
    with pytest.raises(expected_error):
        create_split_dirs(
            all_pdf_dir=mock_pdf_directory,
            target_dir_base=mock_target_directory,
            split_metadata_file=mock_metadata_file,
            split=split
        )
