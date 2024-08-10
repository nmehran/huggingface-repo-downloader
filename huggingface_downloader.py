#!/usr/bin/env python3

# Copyright 2024-present Nima Mehrani
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

"""
Hugging Repository Downloader

This script downloads repositories from Hugging Face, utilizing fast transfer mode when available.
It supports resuming interrupted downloads and skips already downloaded files.

Usage:
    python huggingface_downloader.py <repository_url> [options]

Example:
    (    # TOKEN CAN BE ACCESSED VIA ${HOME}/.cache/huggingface/token)
    python huggingface_downloader.py meta-llama/Meta-Llama-3.1-8B-Instruct/tree/main --use-auth-token

Options:
    -o, --output         Specify output directory (optional)
    --use-auth-token     Use the Hugging Face auth token for private repos
    --revision           Specific revision to download (default: main)
    --help               Show this help message and exit

Fast Transfer Mode:
    To enable fast transfer mode, install the 'hf_transfer' package:
    pip install hf_transfer

    The script will automatically use fast transfer mode if it's installed.


Authentication:
    To set up authentication for private repositories:
    1. Visit https://huggingface.co/settings/tokens
    2. Generate a token
    3. Run 'huggingface-cli login'
    4. Enter your token when prompted

    For Git operations, configure your credentials:
    git config --global credential.helper store
    git credential approve <<EOF
    protocol=https
    host=huggingface.co
    username=_token
    password=YOUR_TOKEN_HERE
    EOF

    Your token is stored at ${HOME}/.cache/huggingface/token
"""

import os

# Initially disable fast transfer
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

import argparse
import hashlib
import importlib
import logging
import re
import sys
import time
from urllib.parse import urlparse

from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global HfApi instance
api = None


def import_huggingface_hub():
    global HfApi, HfFolder, hf_hub_url, hf_hub_download
    global EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError, HfHubHTTPError
    from huggingface_hub import hf_hub_url, hf_hub_download, HfApi, HfFolder
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError, HfHubHTTPError

def enable_fast_transfer():
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
    # Reload necessary modules
    importlib.reload(importlib.import_module('huggingface_hub'))


def initialize_huggingface_hub(use_fast):
    global api

    if use_fast:
        enable_fast_transfer()

    import_huggingface_hub()
    api = HfApi()

    fast_transfer_enabled = is_fast_transfer_enabled()
    if fast_transfer_enabled:
        logger.info("Fast transfer mode is enabled and active")
    elif use_fast:
        logger.warning("Fast transfer mode was requested but 'hf_transfer' is not installed. Falling back to standard transfer.")
    else:
        logger.info("Fast transfer mode is not enabled. Use --fast flag and install 'hf_transfer' for faster downloads.")

    return


def print_auth_instructions():
    instructions = """
To set up authentication for Hugging Face:

1. Visit https://huggingface.co/settings/tokens to generate a token.
2. Run 'huggingface-cli login' and enter your token when prompted.
3. To use the token for Git operations, run the following commands:
   git config --global credential.helper store
   git credential approve <<EOF
   protocol=https
   host=huggingface.co
   username=_token
   password=YOUR_TOKEN_HERE
   EOF
   Replace YOUR_TOKEN_HERE with your actual token.

This will allow you to interact with private repositories and use Git LFS features.

Note: Your token is stored at ${HOME}/.cache/huggingface/token
"""
    print(instructions.strip())


def is_fast_transfer_enabled():
    """Check if fast transfer mode is enabled."""
    if os.environ.get('HF_HUB_ENABLE_HF_TRANSFER') != '1':
        return False

    try:
        importlib.import_module('hf_transfer')
        return True
    except ImportError:
        return False


def get_repo_info(repo_id, revision=None, use_auth_token=None):
    try:
        # Get repository information
        repo_info = api.repo_info(repo_id, revision=revision, token=use_auth_token)

        # Determine the repository type based on available attributes
        if repo_info.pipeline_tag is not None:
            repo_type = "models"
        elif "dataset_infos.json" in [file.rfilename for file in repo_info.siblings]:
            repo_type = "datasets"
        elif "app.py" in [file.rfilename for file in repo_info.siblings]:
            repo_type = "spaces"
        else:
            repo_type = "unknown"

        return True, repo_type

    except RepositoryNotFoundError:
        return False, "Repository not found"
    except HfHubHTTPError as e:
        return False, f"HTTP error: {str(e)}"
    except Exception as e:
        return False, f"An unexpected error occurred: {str(e)}"


def parse_repo_id(url, use_auth_token=None):
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    path = re.sub(r'^huggingface\.co/', '', path)
    path = re.sub(r'/tree/[^/]+$', '', path)
    parts = path.split('/')

    if len(parts) > 2:
        raise ValueError("Invalid repository format. Please use 'owner/repo-name' format.")

    repo_id = '/'.join(parts)
    is_valid, repo_type = get_repo_info(repo_id, use_auth_token=use_auth_token)

    if not is_valid:
        raise ValueError(f"Invalid or inaccessible repository: {repo_id}")

    return f"{repo_type}--{repo_id.replace('/', '--')}"


def get_file_hash(file_path):
    """Calculate the SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def download_file(repo_id, filename, output_dir, use_auth_token, revision="main"):
    try:
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            force_download=True,
            resume_download=True,
            token=use_auth_token
        )
    except Exception as e:
        if "416 Client Error: Requested Range Not Satisfiable" in str(e):
            logger.warning(f"Resume download failed for {filename}. Attempting full download.")
            return hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                local_dir=output_dir,
                local_dir_use_symlinks=False,
                force_download=True,
                resume_download=False,
                token=use_auth_token
            )
        else:
            raise


def download_repo(repo_id, output_dir=None, use_auth_token=None, revision="main"):
    # Extract the original repo name and type from our custom format
    repo_parts = repo_id.split('--')
    repo_type = repo_parts[0]
    original_repo_id = '--'.join(repo_parts[1:]).replace('--', '/')

    try:
        files = api.list_repo_files(original_repo_id, revision=revision, token=use_auth_token)
    except Exception as e:
        logger.error(f"Error listing files in repository {original_repo_id} at revision {revision}: {str(e)}")
        return

    for file in tqdm(files, desc=f"Downloading {repo_type} files (revision: {revision})", unit="file"):
        try:
            local_file = download_file(original_repo_id, file, output_dir, use_auth_token, revision)
            logger.info(f"Downloaded: {local_file}")
        except Exception as e:
            logger.error(f"Error downloading {file}: {str(e)}")
            logger.info("Retrying download...")
            try:
                local_file = download_file(original_repo_id, file, output_dir, use_auth_token, revision)
                logger.info(f"Successfully downloaded on retry: {local_file}")
            except Exception as e:
                logger.error(f"Failed to download {file} after retry: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Download files from Hugging Face.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url", help="URL or ID of the Hugging Face repository")
    parser.add_argument("-o", "--output", help="Output directory (default: script directory)")
    parser.add_argument("--use-auth-token", action="store_true",
                        help="Use the Hugging Face auth token for private repos")
    parser.add_argument("--revision", default="main", help="Specific revision to download (default: main)")
    parser.add_argument("--auth-help", action="store_true", help="Show instructions for setting up authentication")
    parser.add_argument("--fast", action="store_true", help="Enable fast transfer mode using hf_transfer (if installed)")
    args = parser.parse_args()

    initialize_huggingface_hub(args.fast)

    if args.auth_help:
        print_auth_instructions()
        sys.exit(0)

    use_auth_token = HfFolder.get_token() if args.use_auth_token else None

    try:
        repo_id = parse_repo_id(args.url, use_auth_token)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    # Use script directory as default if no output directory is specified
    if args.output:
        base_output_dir = args.output
    else:
        base_output_dir = os.path.dirname(os.path.abspath(__file__))

    output_dir = os.path.join(base_output_dir, repo_id)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Downloading repository: {repo_id}")
    logger.info(f"Revision: {args.revision}")
    logger.info(f"Output directory: {output_dir}")

    try:
        download_repo(repo_id, output_dir, use_auth_token, args.revision)
        logger.info(f"Download completed. Files are stored in: {output_dir}")
    except KeyboardInterrupt:
        logger.info("Download interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")