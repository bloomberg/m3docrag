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
from unittest.mock import patch, MagicMock
from pathlib import Path
import jsonlines
from m3docvqa.downloader import _download_wiki_page, download_wiki_page


@pytest.fixture
def test_urls_and_paths(tmp_path):
    """Fixture to provide sample URLs and save paths for testing."""
    urls = ["https://en.wikipedia.org/wiki/SamplePage1", "https://en.wikipedia.org/wiki/SamplePage2"]
    save_paths = [str(tmp_path / "sample1.pdf"), str(tmp_path / "sample2.pdf")]
    return urls, save_paths


@patch("m3docvqa.downloader.sync_playwright")
def test__download_wiki_page_pdf(mock_playwright, tmp_path):
    """Test downloading a single page as a PDF."""
    url = "https://en.wikipedia.org/wiki/SamplePage"
    save_path = tmp_path / "sample.pdf"
    args = (0, 1, url, str(save_path), 'pdf', 0)

    # Mock Playwright behavior
    mock_browser = MagicMock()
    mock_context = MagicMock()
    mock_page = MagicMock()
    mock_playwright.return_value.__enter__.return_value.chromium.launch.return_value = mock_browser
    mock_browser.new_context.return_value = mock_context
    mock_context.new_page.return_value = mock_page

    # Call the function
    downloaded, error = _download_wiki_page(args)

    # Assertions
    assert downloaded is True
    assert error is None
    mock_page.goto.assert_called_once_with(url)
    mock_page.pdf.assert_called_once_with(path=str(save_path))


@patch("m3docvqa.downloader.sync_playwright")
def test__download_wiki_page_png(mock_playwright, tmp_path):
    """Test downloading a single page as a PNG."""
    url = "https://en.wikipedia.org/wiki/SamplePage"
    save_path = tmp_path / "sample.png"
    args = (0, 1, url, str(save_path), 'png', 0)

    # Mock Playwright behavior
    mock_browser = MagicMock()
    mock_context = MagicMock()
    mock_page = MagicMock()
    mock_playwright.return_value.__enter__.return_value.chromium.launch.return_value = mock_browser
    mock_browser.new_context.return_value = mock_context
    mock_context.new_page.return_value = mock_page

    # Call the function
    downloaded, error = _download_wiki_page(args)

    # Assertions
    assert downloaded is True
    assert error is None
    mock_page.goto.assert_called_once_with(url)
    mock_page.screenshot.assert_called_once_with(path=str(save_path), full_page=True)


@patch("m3docvqa.downloader._download_wiki_page")
def test_download_wiki_page_batch(mock_download_wiki_page, tmp_path, test_urls_and_paths):
    """Test batch downloading multiple Wikipedia pages."""
    urls, save_paths = test_urls_and_paths
    result_jsonl_path = tmp_path / "download_results.jsonl"

    # Mock individual downloads to always succeed
    mock_download_wiki_page.side_effect = [(True, None), (True, None)]

    # Call the function
    results = download_wiki_page(urls, save_paths, 'pdf', str(result_jsonl_path), proc_id=0, n_proc=1)

    # Assertions
    assert results == [True, True]
    assert result_jsonl_path.exists()

    # Check JSONL log entries
    with jsonlines.open(result_jsonl_path, 'r') as reader:
        log_entries = list(reader)
        assert len(log_entries) == 2
        assert log_entries[0]['downloaded'] is True
        assert log_entries[0]['error'] is None
        assert log_entries[1]['downloaded'] is True
        assert log_entries[1]['error'] is None
