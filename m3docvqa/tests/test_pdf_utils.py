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
from m3docvqa.pdf_utils import is_pdf_downloaded, is_pdf_clean, get_images_from_pdf
from pathlib import Path
from PIL import Image
from reportlab.pdfgen import canvas  # For creating sample PDFs


@pytest.fixture
def sample_pdf(tmp_path) -> Path:
    """Create a temporary sample PDF file for testing."""
    pdf_path = tmp_path / "sample.pdf"
    c = canvas.Canvas(str(pdf_path))
    c.drawString(100, 100, "Sample PDF text for testing.")  # Add sample text to the PDF
    c.save()
    return pdf_path


@pytest.fixture
def corrupted_pdf(tmp_path) -> Path:
    """Create a temporary, corrupted PDF file for testing."""
    pdf_path = tmp_path / "corrupted.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 corrupted content")  # Write incomplete/corrupted PDF content
    return pdf_path


def test_is_pdf_downloaded_existing_pdf(sample_pdf):
    """Test is_pdf_downloaded on a valid, existing PDF."""
    assert is_pdf_downloaded(str(sample_pdf)) is True, "Expected PDF to be recognized as downloaded."


def test_is_pdf_downloaded_nonexistent_pdf(tmp_path):
    """Test is_pdf_downloaded on a non-existent PDF file."""
    non_existent_pdf = tmp_path / "non_existent.pdf"
    assert is_pdf_downloaded(str(non_existent_pdf)) is False, "Expected non-existent PDF to be marked as not downloaded."


def test_is_pdf_clean_valid_pdf(sample_pdf):
    """Test is_pdf_clean on a valid, clean PDF."""
    assert is_pdf_clean(str(sample_pdf)) is True, "Expected PDF to be recognized as clean."


def test_is_pdf_clean_corrupted_pdf(corrupted_pdf):
    """Test is_pdf_clean on a corrupted PDF."""
    assert is_pdf_clean(str(corrupted_pdf)) is False, "Expected corrupted PDF to be marked as not clean."


def test_get_images_from_pdf_extract_images(sample_pdf, tmp_path):
    """Test get_images_from_pdf to ensure it extracts images correctly."""
    image_dir = tmp_path / "images"
    images = get_images_from_pdf(str(sample_pdf), save_dir=str(image_dir), dpi_resolution=72, save_type='png')

    # Verify that at least one image was extracted
    assert len(images) > 0, "Expected at least one image to be extracted from the PDF."

    # Verify that images were saved to the directory
    saved_images = list(image_dir.glob("*.png"))
    assert len(saved_images) == len(images), "Expected number of saved images to match the number of extracted images."

    # Verify that the saved image files exist and are valid
    for image_path in saved_images:
        with Image.open(image_path) as img:
            assert img.format == "PNG", "Expected saved image to be in PNG format."


def test_get_images_from_pdf_no_save_dir(sample_pdf):
    """Test get_images_from_pdf without saving images, only returning them as a list."""
    images = get_images_from_pdf(str(sample_pdf), save_dir=None, dpi_resolution=72)
    assert len(images) > 0, "Expected at least one image to be returned without saving."
    assert all(isinstance(image, Image.Image) for image in images), "Expected all returned items to be PIL Image objects."

