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

"""
Downloader Module for M3DocVQA

This module provides functions to download Wikipedia pages in either PDF or PNG format
for the M3DocVQA dataset. It uses Playwright to load and capture the pages in a headless
browser environment and saves each page in the specified format.

Functions:
    - _download_wiki_page: Downloads a single Wikipedia page as a PDF or PNG.
    - download_wiki_page: Manages the downloading of multiple Wikipedia pages.
"""

from playwright.sync_api import sync_playwright
from loguru import logger
from pathlib import Path
import jsonlines
from tqdm.auto import tqdm
from m3docvqa.pdf_utils import is_pdf_downloaded


def _download_wiki_page(args: tuple[int, int, str, str, str, int]) -> tuple[bool, Exception | None]:
    """Download a single Wikipedia page as a PDF or PNG using Playwright.

    Args:
        args (Tuple[int, int, str, str, str, int]): Contains order in batch, total count, URL, save path,
            save type ('pdf' or 'png'), and process ID.

    Returns:
        Tuple[bool, Optional[Exception]]: A tuple where the first element is a boolean indicating success,
            and the second element is an exception if an error occurred, or None otherwise.
    """
    order_i, total, url, save_path, save_type, proc_id = args

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(ignore_https_errors=True)
            page = context.new_page()
            page.set_default_timeout(30000)  # 30 seconds timeout

            page.goto(url)
            if save_type == 'png':
                page.screenshot(path=save_path, full_page=True)
            elif save_type == 'pdf':
                page.emulate_media(media="screen")
                page.pdf(path=save_path)

            browser.close()

        return True, None
    except Exception as error:
        logger.warning(f"Failed to download {url} as {save_type}. Error: {error}")
        return False, error


def download_wiki_page(
    urls: list[str],
    save_paths: list[str],
    save_type: str,
    result_log_dir: str,
    proc_id: int = 0,
    n_proc: int = 1
) -> list[bool]:
    """Download multiple Wikipedia pages and log progress.

    Args:
        urls (List[str]): List of Wikipedia URLs to download.
        save_paths (List[str]): List of paths where each downloaded file will be saved.
        save_type (str): File type to save each page as ('pdf' or 'png').
        result_log_dir (str): Path to the directory where the download results will be logged.
        proc_id (int, optional): Process ID for parallel processing. Defaults to 0.
        n_proc (int, optional): Total number of processes running in parallel. Defaults to 1.

    Returns:
        List[bool]: A list of booleans indicating whether each download was successful.
    """
    total = len(urls)
    all_args = [(i, total, url, str(save_path), save_type, proc_id) 
                for i, (url, save_path) in enumerate(zip(urls, save_paths))]

    # create log directory if it doesn't exist
    log_dir = Path(result_log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(total=len(all_args), ncols=100, disable=not (proc_id == 0))

    results = []
    n_downloaded = 0

    for args in all_args:
        downloaded, error = _download_wiki_page(args)

        if downloaded:
            n_downloaded += 1

        pbar.set_description(f"Process: {proc_id}/{n_proc} - Downloaded: {n_downloaded}/{total}")
        pbar.update(1)

        results.append(downloaded)
        
        # Write to process-specific log file
        proc_result_path = log_dir / f'process_{proc_id}_{n_proc}.jsonl'
        with jsonlines.open(proc_result_path, mode='a') as writer:  
            writer.write({
                'downloaded': downloaded,
                'args': [arg if not isinstance(arg, Path) else str(arg) for arg in args],
                'error': str(error) if error else None
            })

    pbar.close()
    return results
