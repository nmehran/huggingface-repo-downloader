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
    python huggingface_downloader.py meta-llama/Meta-Llama-3.1-8B-Instruct/tree/main --use-auth-token

Options:
    -o, --output         Specify output directory (optional)
    --use-auth-token     Use the Hugging Face auth token for private repos
    --revision           Specific revision to download (default: main)
    --fast               Enable fast transfer mode (requires hf_transfer package)
    --auth-help          Show instructions for setting up authentication
    --ignore-file        Specify a file containing glob patterns to ignore
    --help               Show this help message and exit

Fast Transfer Mode:
    To enable fast transfer mode, install the 'hf_transfer' package:
    pip install hf_transfer

    Use the --fast flag to enable fast transfer mode.

Authentication:
    To set up authentication for private repositories:
    1. Visit https://huggingface.co/settings/tokens
    2. Generate a token
    3. Run 'huggingface-cli login'
    4. Enter your token when prompted

    For detailed instructions, use the --auth-help flag.
"""

import os

# Initially disable fast transfer
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

import argparse
import importlib
import logging
import re
import sys
import time
from typing import Tuple, List, Optional
from urllib.parse import urlparse
from fnmatch import fnmatch
from tqdm import tqdm


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global HfApi instance
api: Optional['HfApi'] = None


def import_huggingface_hub(use_fast: bool):
    """
    Import necessary modules and functions from huggingface_hub.

    This function imports required classes and functions from the huggingface_hub library
    and makes them available globally, after fast transfer is set.

    Args:
        use_fast (bool): Whether to enable fast transfer mode.
    """
    if use_fast:
        # Set the environment variable to enable fast transfer mode, prior to importing huggingface_hub
        os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

    global HfApi, HfFolder, hf_hub_url, hf_hub_download
    global EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError, HfHubHTTPError
    from huggingface_hub import hf_hub_url, hf_hub_download, HfApi, HfFolder
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError, HfHubHTTPError


def initialize_huggingface_hub(use_fast: bool):
    """
    Initialize the Hugging Face Hub API and set up fast transfer mode if requested.

    Args:
        use_fast (bool): Whether to enable fast transfer mode.
    """
    global api

    import_huggingface_hub(use_fast)

    api = HfApi()

    fast_transfer_enabled = is_fast_transfer_enabled()
    if fast_transfer_enabled:
        logger.info("Fast transfer mode is enabled and active")
    elif use_fast:
        logger.warning("Fast transfer mode was requested but 'hf_transfer' is not installed. "
                       "Falling back to standard transfer.")
    else:
        logger.info("Fast transfer mode is not enabled. "
                    "Use --fast flag and install 'hf_transfer' if available bandwidth is greater than 500MB/s.")


def print_auth_instructions():
    """
    Print instructions for setting up authentication with Hugging Face.

    This function displays step-by-step instructions for generating and configuring
    a Hugging Face authentication token.
    """
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


def is_fast_transfer_enabled() -> bool:
    """
    Check if fast transfer mode is enabled.

    Returns:
        bool: True if fast transfer mode is enabled and the hf_transfer module is installed, False otherwise.
    """
    if os.environ.get('HF_HUB_ENABLE_HF_TRANSFER') != '1':
        return False

    try:
        importlib.import_module('hf_transfer')
        return True
    except ImportError:
        return False


def should_ignore_file(file: str, ignore_patterns: List[str]) -> bool:
    """
    Check if a file should be ignored based on the given ignore patterns.

    Args:
        file (str): The name of the file to check.
        ignore_patterns (List[str]): A list of glob patterns to match against.

    Returns:
        bool: True if the file should be ignored, False otherwise.
    """
    return any(fnmatch(file, pattern) for pattern in ignore_patterns)


def read_ignore_file(ignore_file: Optional[str]) -> List[str]:
    """
    Read and parse an ignore file, returning a list of ignore patterns.

    Args:
        ignore_file (Optional[str]): The path to the ignore file.

    Returns:
        List[str]: A list of ignore patterns, or an empty list if the file doesn't exist or is empty.
    """
    if not ignore_file:
        return []
    try:
        with open(ignore_file, 'r') as f:
            return [line for line in f if (stripped := line.strip()) and not stripped.startswith('#')]
    except IOError as e:
        logger.error(f"Error reading ignore file {ignore_file}: {str(e)}")
        return []


def get_repo_info(
        repo_id: str,
        revision: Optional[str] = None,
        use_auth_token: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Retrieve information about a Hugging Face repository.

    Args:
        repo_id (str): The ID of the repository.
        revision (Optional[str]): The specific revision to query (default: None).
        use_auth_token (Optional[str]): The authentication token to use (default: None).

    Returns:
        Tuple[bool, str]: A tuple containing a boolean indicating success and a string with the repo type or error message.
    """
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
        return False, f"Repository `{repo_id}` not found"
    except RevisionNotFoundError:
        return False, f"Revision `{revision}` not found"
    except HfHubHTTPError as e:
        return False, f"HTTP error: {str(e)}"
    except Exception as e:
        return False, f"An unexpected error occurred: {str(e)}"


def process_repo_info(
        url: str,
        cli_revision: Optional[str],
        use_auth_token: Optional[str]
) -> Tuple[str, str, str, str]:
    """
    Process repository information and validate the repository.

    Args:
        url (str): The URL or ID of the Hugging Face repository.
        cli_revision (Optional[str]): The revision specified in the CLI arguments.
        use_auth_token (Optional[str]): The authentication token to use.

    Returns:
        Tuple[str, str, str, str]: A tuple containing the repository ID, formatted repository ID, repository type, and revision.

    Raises:
        SystemExit: If the repository is invalid or inaccessible.
    """
    try:
        repo_id, url_revision = parse_repo_id(url)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    # Use CLI revision if provided, otherwise use URL revision, fallback to 'main'
    revision = cli_revision or url_revision or "main"

    # Validate the repository
    is_valid, repo_type = get_repo_info(repo_id, revision=revision, use_auth_token=use_auth_token)
    if not is_valid:
        logger.error(f"Invalid or inaccessible repository: {repo_id}")
        sys.exit(1)

    formatted_repo_id = f"{repo_type}--{repo_id.replace('/', '--')}"

    return repo_id, formatted_repo_id, repo_type, revision


def parse_repo_id(url: str) -> Tuple[str, Optional[str]]:
    """
    Parse a Hugging Face repository URL or ID to extract the repository ID and revision.

    Args:
        url (str): The URL or ID of the Hugging Face repository.

    Returns:
        Tuple[str, Optional[str]]: A tuple containing the repository ID and revision (if specified).

    Raises:
        ValueError: If the repository format is invalid.
    """
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    path = re.sub(r'^huggingface\.co/', '', path)

    # Extract revision if present
    revision = None
    tree_match = re.search(r'/tree/([^/]+)$', path)
    if tree_match:
        revision = tree_match.group(1)
        path = re.sub(r'/tree/[^/]+$', '', path)

    parts = path.split('/')

    if len(parts) > 2:
        raise ValueError("Invalid repository format. Please use 'owner/repo-name' format.")

    repo_id = '/'.join(parts)
    return repo_id, revision


def download_file(
        repo_id: str,
        filename: str,
        output_dir: str,
        use_auth_token: Optional[str],
        revision: str = "main"
) -> str:
    """
    Download a single file from a Hugging Face repository.

    Args:
        repo_id (str): The ID of the repository.
        filename (str): The name of the file to download.
        output_dir (str): The directory to save the downloaded file.
        use_auth_token (Optional[str]): The authentication token to use.
        revision (str): The specific revision to download from (default: "main").

    Returns:
        str: The path to the downloaded file.

    Raises:
        SystemExit: If authentication fails.
        Exception: For other download errors.
    """
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
        elif "401 Client Error" in str(e) and "Cannot access gated repo" in str(e):
            logger.error("Error: Cannot access gated repository. "
                         "Use --auth-help for instructions on setting up authentication.")
            logger.error("If you have already set up authentication, make sure you're using the --use-auth-token flag.")
            sys.exit(1)
        else:
            raise


def download_repo(
        repo_id: str,
        repo_type: str,
        output_dir: Optional[str] = None,
        use_auth_token: Optional[str] = None,
        revision: str = "main",
        ignore_patterns: Optional[List[str]] = None
):
    """
    Download all files from a Hugging Face repository.

    Args:
        repo_id (str): The ID of the repository.
        repo_type (str): The type of the repository (e.g., "models", "datasets", "spaces").
        output_dir (Optional[str]): The directory to save the downloaded files (default: None).
        use_auth_token (Optional[str]): The authentication token to use (default: None).
        revision (str): The specific revision to download from (default: "main").
        ignore_patterns (Optional[List[str]]): A list of glob patterns for files to ignore (default: None).
    """
    try:
        files = api.list_repo_files(repo_id, revision=revision, token=use_auth_token)
    except Exception as e:
        logger.error(f"Error listing files in repository {repo_id} at revision {revision}: {str(e)}")
        return

    for file in tqdm(files, desc=f"Downloading {repo_type} files (revision: {revision})", unit="file"):
        if ignore_patterns and should_ignore_file(file, ignore_patterns):
            logger.info(f"Skipping ignored file: {file}")
            continue
        try:
            local_file = download_file(repo_id, file, output_dir, use_auth_token, revision)
            logger.info(f"Downloaded: {local_file}")
        except Exception as e:
            logger.error(f"Error downloading {file}: {str(e)}")
            logger.info("Retrying download...")
            try:
                local_file = download_file(repo_id, file, output_dir, use_auth_token, revision)
                logger.info(f"Successfully downloaded on retry: {local_file}")
            except Exception as e:
                logger.error(f"Failed to download {file} after retry: {str(e)}")


def main():
    """
    Main function to handle command-line arguments and orchestrate the download process.
    """
    parser = argparse.ArgumentParser(description="Download files from Hugging Face.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # Command-line Parameters
    parser.add_argument("url",
                        help="URL or ID of the Hugging Face repository")
    parser.add_argument("-o", "--output",
                        help="Output directory (default: script directory)")
    parser.add_argument("--use-auth-token",
                        action="store_true",
                        help="Use the Hugging Face auth token for private repos")
    parser.add_argument("--revision",
                        default="main",
                        help="Specific revision to download (default: main)")
    parser.add_argument("--auth-help",
                        action="store_true",
                        help="Show instructions for setting up authentication")
    parser.add_argument("--fast",
                        action="store_true",
                        help="Enable fast transfer mode using hf_transfer (if installed)")
    parser.add_argument("--ignore-file",
                        help="Specify a file containing glob patterns to ignore")

    args = parser.parse_args()

    if args.auth_help:
        print_auth_instructions()
        sys.exit(0)

    # Initialize `HfApi` and import `hf_hub_download` based on whether fast transfer mode is used
    initialize_huggingface_hub(args.fast)

    use_auth_token = HfFolder.get_token() if args.use_auth_token else None

    # Extract directory format, infer repository type, and get revision
    repo_id, formatted_repo_id, repo_type, revision = process_repo_info(args.url, args.revision, use_auth_token)

    # Use script directory as default if no output directory is specified
    if args.output:
        base_output_dir = args.output
    else:
        base_output_dir = os.path.dirname(os.path.abspath(__file__))

    output_dir = os.path.join(base_output_dir, formatted_repo_id)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Downloading repository: {repo_id}")
    logger.info(f"Revision: {args.revision}")
    logger.info(f"Output directory: {output_dir}")

    ignore_patterns = read_ignore_file(args.ignore_file)
    if ignore_patterns:
        logger.info(f"Using ignore patterns from {args.ignore_file}")

    # Main download function
    try:
        download_repo(repo_id, repo_type, output_dir, use_auth_token, args.revision, ignore_patterns)
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