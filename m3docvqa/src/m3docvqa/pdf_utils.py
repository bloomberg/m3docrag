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

"""PDF Utilities Module for M3DocVQA.

This module provides utility functions for managing and processing PDF files in the M3DocVQA dataset.
It includes functions for checking if a PDF has been downloaded, verifying if a PDF is clean (not corrupted),
and extracting images from PDF pages.

Functions:
    - is_pdf_downloaded: Checks if a given PDF file exists and can be opened without errors.
    - is_pdf_clean: Checks if a PDF file is clean (not corrupted) and can be read without issues.
    - get_images_from_pdf: Extracts images from each page of a PDF and optionally saves them in a specified directory.
"""

from pdf2image import convert_from_path
from PIL import Image
from pdfrw import PdfReader
from pathlib import Path
from io import BytesIO
from loguru import logger


def is_pdf_downloaded(pdf_path: str) -> bool:
    """Check if the PDF file exists and can be opened without errors.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        bool: True if the PDF file is downloaded and accessible; False otherwise.
    """
    try:
        with open(pdf_path, "rb") as f:
            f.read(1)  # Attempt to read a byte to verify file exists and is accessible
        return True
    except Exception as e:
        logger.trace(f"Failed to open PDF at {pdf_path}: {e}")
        return False


def is_pdf_clean(pdf_path: str) -> bool:
    """Verify if a PDF file is clean (not corrupted) and can be read without errors.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        bool: True if the PDF file is clean and readable; False otherwise.
    """
    try:
        with open(pdf_path, "rb") as f:
            idata = f.read()
        ibuffer = BytesIO(idata)
        PdfReader(ibuffer)  # Attempt to read the PDF structure for validity
        return True
    except Exception as error:
        logger.warning(f"PDF at {pdf_path} is corrupted or unreadable: {error}")
        return False


def get_images_from_pdf(
    pdf_path: str,
    save_dir: str = None,
    max_pages: int = None,
    dpi_resolution: int = 144,
    save_type: str = 'png'
) -> list[Image.Image]:
    """Extract images from each page of a PDF and optionally save them to a directory.

    Args:
        pdf_path (str): Path to the PDF file.
        save_dir (str, optional): Directory where images will be saved. If None, images are not saved. Defaults to None.
        max_pages (int, optional): Maximum number of pages to process. If None, all pages are processed. Defaults to None.
        dpi_resolution (int, optional): Resolution for image extraction. Defaults to 144.
        save_type (str, optional): Image file type to save as ('png', 'jpg', etc.). Defaults to 'png'.

    Returns:
        list[Image.Image]: A list of images extracted from each page of the PDF.
    """
    pdf_path_obj = Path(pdf_path)
    assert pdf_path_obj.exists(), f"PDF file {pdf_path} does not exist."

    out_images = []

    # Create save directory if saving images is enabled
    if save_dir:
        save_dir_path = Path(save_dir)
        save_dir_path.mkdir(exist_ok=True, parents=True)

    try:
        # Convert PDF to images using pdf2image
        images = convert_from_path(pdf_path, dpi=dpi_resolution)
        logger.info(f"PDF {pdf_path} has {len(images)} pages.")

        # Limit the number of pages processed if max_pages is set
        if max_pages:
            images = images[:max_pages]

        for page_index, image in enumerate(images):
            out_images.append(image)

            # Save image if save directory is specified
            if save_dir:
                save_page_path = save_dir_path / f"{pdf_path_obj.stem}_{page_index + 1}.{save_type}"
                if not save_page_path.exists():
                    image.save(save_page_path)
                    logger.info(f"Saved page {page_index + 1} as image at {save_page_path}")

    except Exception as e:
        logger.error(f"Error extracting images from PDF {pdf_path}: {e}")

    return out_images
