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
from loguru import logger
from pdf2image import convert_from_path
from collections import Counter


def get_images_from_pdf(pdf_path, max_pages=None, dpi_resolution=144, save_dir='/tmp/', save_image=False, save_type='png', verbose=False):
    pdf_path = Path(pdf_path)
    assert pdf_path.exists(), f"{pdf_path} does not exist"

    pdf_fname = pdf_path.name

    images = convert_from_path(pdf_path, dpi=dpi_resolution)

    # PIL.PpmImagePlugin.PpmImageFile -> PIL.Image.Image
    images = [img.convert('RGB') for img in images]
    
    # resizing to the most common image size so that we can stack in pytorch tensor
    # PDFs (e.g., MMLongBench-Doc) have different image sizes
    # width=1,224 and height=1,584
    # 1440, 810
    # 1191, 1684
    # 1440, 1080
    # 1536, 1152
    
    # 1) find the most common image size
    img_size_counter = Counter()
    for img in images:
        img_size_counter[img.size] += 1
    common_img_size, common_img_size_count = img_size_counter.most_common(1)[0]

    # 2) if pages have different sizes -> resize all pages to that image size
    if len(images) != common_img_size_count:
        logger.info(f"total: {len(images)} pages")
        logger.info(f"resizing to the most common image size: {common_img_size} with count: {common_img_size_count}")
        images = [img.resize(common_img_size) for img in images]

    if save_image:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

        for page_index, page_image in enumerate(images):
            save_page_path = save_dir / f"{pdf_fname}_{page_index+1}.{save_type}"
            if not save_page_path.exists():
                page_image.save(save_page_path)
                if verbose:
                    logger.info(f"Page {page_index} saved at {save_page_path}")

    return images



if __name__ == '__main__':
    get_images_from_pdf(
        pdf_path="./multimodalqa_screenshots_pdfs/0df5cc80bcd2a27b91224d658ad3a7b5.pdf",
        save_dir='./tmp/',
        save_image=True
    )
