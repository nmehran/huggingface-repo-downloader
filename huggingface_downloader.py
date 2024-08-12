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
    --url-file           Specify a file containing repository URLs to download
    --force              Force re-download of all files, ignoring existing metadata
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

Metadata Handling:
    The script uses metadata to keep track of downloaded files and their versions.
    This helps to avoid unnecessary re-downloads of unchanged files.
    Use the --force flag to override this behavior and re-download all files.
"""

import os

# Initially disable fast transfer
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

import argparse
import hashlib
import importlib
import json
import logging
import re
import sys
import time
from fnmatch import fnmatch
from tqdm import tqdm
from typing import Tuple, List, Dict, Optional
from urllib.parse import unquote, urlparse

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

    global HfApi, HfFolder, hf_hub_download
    global DatasetInfo, ModelInfo, SpaceInfo, RepoFile
    global EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError, HfHubHTTPError
    from huggingface_hub import hf_hub_download, HfApi, HfFolder
    from huggingface_hub.hf_api import DatasetInfo, ModelInfo, SpaceInfo, RepoFile
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

    if is_fast_transfer_enabled():
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
        repo_type: str,
        revision: Optional[str] = None,
        use_auth_token: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Retrieve information about a Hugging Face repository.

    Args:
        repo_id (str): The ID of the repository.
        repo_type (str): The type of the repository (e.g., "model", "dataset", "space").
        revision (Optional[str]): The specific revision to query (default: None).
        use_auth_token (Optional[str]): The authentication token to use (default: None).

    Returns:
        Tuple[bool, str]: A tuple containing a boolean indicating success and a string with the repo type or error message.
    """
    if repo_type is None:
        # Try model first, then dataset if not found
        repo_types_to_try = ["model", "dataset", "space"]
    else:
        repo_types_to_try = [repo_type]

    for rt in repo_types_to_try:
        try:
            repo_info = api.repo_info(repo_id, repo_type=rt, revision=revision, token=use_auth_token)
            if isinstance(repo_info, (DatasetInfo, ModelInfo, SpaceInfo)):
                return True, rt
        except RepositoryNotFoundError:
            continue
        except RevisionNotFoundError:
            return False, f"Revision `{revision}` not found"
        except HfHubHTTPError as e:
            return False, f"HTTP error: {str(e)}"
        except Exception as e:
            return False, f"An unexpected error occurred: {str(e)}"

    return False, f"Repository `{repo_id}` not found or not a valid model or dataset"


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
        repo_id, repo_type, url_revision = parse_repo_id(url)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    # Use CLI revision if provided, otherwise use URL revision, fallback to 'main'
    revision = cli_revision or url_revision or "main"

    # Validate the repository
    is_valid, repo_type = get_repo_info(repo_id, repo_type, revision=revision, use_auth_token=use_auth_token)
    if not is_valid:
        logger.error(f"Invalid or inaccessible repository: {repo_id}")
        sys.exit(1)

    formatted_repo_id = f"{repo_type}--{repo_id.replace('/', '--')}"

    if revision != 'main':
        formatted_repo_id = f"{formatted_repo_id}_{revision}"

    return repo_id, formatted_repo_id, repo_type, revision


def parse_repo_id(url: str) -> Tuple[str, Optional[str], str]:
    """
    Parse a Hugging Face repository URL or ID to extract the repository ID, revision, and type.

    Args:
        url (str): The URL or ID of the Hugging Face repository.

    Returns:
        Tuple[str, Optional[str], str]: A tuple containing the repository ID, revision (if specified), and repository type.

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

    # Determine repo_type based on URL structure
    if len(parts) > 2:
        repo_type = parts[0]
        repo_id = '/'.join(parts[1:])
    else:
        repo_type = None
        repo_id = '/'.join(parts)

    # Map repository types
    repo_type_mapping = {
        "datasets": "dataset",
        "models": "model",
        "spaces": "space"
    }

    if repo_type is not None:
        repo_type = repo_type_mapping.get(repo_type)
        if repo_type is None:
            raise ValueError(f"Unsupported repository type: {parts[0]}. Must be one of 'models', 'datasets', or 'spaces'")

    return repo_id, repo_type, revision


def get_repo_tree_metadata(repo_id: str, repo_type: str, revision: str, use_auth_token: Optional[str]) -> Optional[Dict[str, Dict]]:
    """
    Fetch metadata for all files in a Hugging Face repository.

    Args:
        repo_id (str): The ID of the repository.
        repo_type (str): The type of the repository (e.g., "model", "dataset", "space").
        revision (str): The specific revision to fetch metadata for.
        use_auth_token (Optional[str]): The authentication token to use.

    Returns:
        Optional[Dict[str, Dict]]: A dictionary containing metadata for each file,
        or None if there was an error fetching the metadata.
    """
    file_metadata = {}
    try:
        for item in api.list_repo_tree(repo_id=repo_id,
                                       repo_type=repo_type,
                                       revision=revision,
                                       recursive=True,
                                       token=use_auth_token):
            if isinstance(item, RepoFile):
                file_metadata[item.path] = {
                    'size': getattr(item, 'size', None),
                    'blob_id': getattr(item, 'blob_id', None),
                    'lfs_sha256': getattr(item.lfs, 'sha256', None) if getattr(item, 'lfs', None) else None
                }
        return file_metadata
    except Exception as e:
        logger.error(f"Error fetching repository tree for {repo_id}: {str(e)}")
        return None


def get_file_hash(file_path):
    """Calculate the SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def should_download_file(file_path: str, current_metadata: Dict, stored_metadata: Dict, output_dir: str) -> bool:
    """
    Determine if a file should be downloaded based on its metadata and actual file size.

    Args:
        file_path (str): The path of the file.
        current_metadata (Dict): The current metadata for the file.
        stored_metadata (Dict): The stored metadata for the file.
        output_dir (str): The directory where the file is or will be stored.

    Returns:
        bool: True if the file should be downloaded, False otherwise.
    """
    if file_path not in stored_metadata:
        return True  # New file, should download

    current = current_metadata.get(file_path, {})
    stored = stored_metadata.get(file_path, {})
    full_path = os.path.join(output_dir, file_path)

    if not os.path.exists(full_path):
        return True  # File doesn't exist on disk, should download

    actual_size = os.path.getsize(full_path)
    if actual_size != current.get('size'):
        return True  # File size on disk differs from current remote repository metadata, should download

    # For LFS files, compare SHA256 hashes
    if current.get('lfs_sha256'):
        if get_file_hash(full_path) != current.get('lfs_sha256'):
            return True  # File hash on disk differs from current metadata, should download
        return current.get('lfs_sha256') != stored.get('lfs_sha256')  # Compare with stored metadata

    # For non-LFS files, compare size and blob_id to current remote repository metadata
    return current.get('size') != stored.get('size') or current.get('blob_id') != stored.get('blob_id')


def load_stored_metadata(metadata_file: Optional[str]) -> Dict:
    """
    Load stored metadata from a file.

    Args:
        metadata_file (Optional[str]): The file to load metadata from.

    Returns:
        Dict: The loaded metadata, or an empty dictionary if loading failed.
    """
    if not metadata_file:
        return {}
    try:
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        logger.warning(f"Error reading metadata file: {str(e)}. Starting with empty metadata.")
    return {}


def store_metadata(metadata_file: Optional[str], metadata: Dict):
    """
    Store metadata to a file.

    Args:
        metadata_file (Optional[str]): The file to store metadata in.
        metadata (Dict): The metadata to store.
    """
    if not metadata_file:
        return
    try:
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
    except IOError as e:
        logger.error(f"Error writing metadata file: {str(e)}")


def get_metadata_file(output_dir: str, formatted_repo_id: str) -> str:
    """
    Generate the path for the metadata file.

    Args:
        output_dir (str): The output directory for downloads.
        formatted_repo_id (str): The formatted repository ID.

    Returns:
        str: The full path to the metadata file.
    """
    cache_directory = os.path.join(output_dir, ".cache", "huggingface_downloader")
    os.makedirs(cache_directory, exist_ok=True)

    return os.path.join(cache_directory, f"{formatted_repo_id}_metadata.json")


def set_output_directory(specified_output: Optional[str], formatted_repo_id: str) -> str:
    """
    Set and create the output directory for downloads.

    This function determines the base output directory (either user-specified or script directory),
    creates a subdirectory for the specific repository, and ensures the directory exists.

    Args:
        specified_output (Optional[str]): The output directory specified by the user, if any.
        formatted_repo_id (str): The formatted repository ID to use as a subdirectory name.

    Returns:
        str: The full path to the created output directory.
    """
    if specified_output:
        base_output_dir = specified_output
    else:
        base_output_dir = os.path.dirname(os.path.abspath(__file__))

    output_dir = os.path.join(base_output_dir, formatted_repo_id)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")
    return output_dir


def validate_and_process_urls(
        args: argparse.Namespace,
        parser: argparse.ArgumentParser
) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Validate the URL and URL file arguments, and process the URLs.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        parser (argparse.ArgumentParser): The argument parser object.

    Returns:
        Tuple[List[str], Dict[str, List[str]]]: A tuple containing:
            - List[str]: List of valid URLs.
            - Dict[str, List[str]]: A result dictionary for tracking successes and failures.
    """
    results = {'success': [], 'failure': []}

    valid_urls = []

    if bool(args.url) == bool(args.url_file):
        parser.print_help()
        logger.error("Error: Exactly one of URL or URL file must be provided.")
        return valid_urls, results

    if args.url_file:
        try:
            valid_urls, invalid_urls = process_url_file(args.url_file)
            results['failure'].extend(invalid_urls)

            if not valid_urls:
                return valid_urls, results

            logger.info(f"Found {len(valid_urls)} valid URLs in {args.url_file}")
        except (FileNotFoundError, IOError) as e:
            logger.error(str(e))
            return valid_urls, results
    else:
        if not is_valid_repo_id_or_url(args.url):
            logger.error(f"Invalid repository URL or ID: {args.url}")
            return valid_urls, results
        valid_urls = [args.url]
        logger.info(f"Validated repository URL/ID: {args.url}")

    return valid_urls, results


def is_valid_repo_id_or_url(input_string: str) -> bool:
    """
    Check if the input string is a valid Hugging Face repository ID or URL.

    Args:
        input_string (str): The string to validate.

    Returns:
        bool: True if the input is a valid repository ID or URL, False otherwise.
    """
    # Decode URL-encoded string
    decoded_input = unquote(input_string).strip().lstrip('/')

    # Regex for repository ID with optional type prefix, tree, and branch
    repo_id_pattern = (
        r'^(?:datasets|spaces)?/?'                                   # Optional prefix for datasets or spaces
        r'([a-zA-Z0-9][-a-zA-Z0-9._]{0,38}[a-zA-Z0-9])/'             # Username
        r'([a-zA-Z0-9][-a-zA-Z0-9._]{1,100}[a-zA-Z0-9])'             # Repository name
        r'(?:/tree/([a-zA-Z0-9][-a-zA-Z0-9._]{0,38}[a-zA-Z0-9]))?$'  # Optional tree and revision branch
    )

    # Regex for full Hugging Face URL
    url_pattern = r'^(?:https?://)?(?:www\.)?huggingface\.co/' + repo_id_pattern.lstrip('^')

    return bool(re.match(repo_id_pattern, decoded_input) or re.match(url_pattern, decoded_input))


def process_url_file(file_path: str) -> Tuple[List[str], List[str]]:
    """
    Read a file containing Hugging Face repository URLs or IDs and return a list of valid entries.

    Args:
        file_path (str): Path to the file containing URLs or repository IDs.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing:
            - List of valid repository URLs or IDs from the file.
            - List of invalid entries from the file.

    Raises:
        FileNotFoundError: If the specified file is not found.
        IOError: If there's an error reading the file.
    """
    try:
        with open(file_path, 'r') as f:
            entries = [line.strip() for line in f if line.strip()]

        if not entries:
            logger.error(f"The file {file_path} is empty.")
            return [], []

        valid_entries = []
        invalid_entries = []
        for entry in entries:
            if is_valid_repo_id_or_url(entry):
                valid_entries.append(entry)
            else:
                logger.warning(f"Skipping invalid URL or Repository-ID: {entry}")
                invalid_entries.append(entry)

        return valid_entries, invalid_entries
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except IOError as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        raise


def download_file(
        repo_id: str,
        repo_type: str,
        filename: str,
        output_dir: str,
        use_auth_token: Optional[str],
        revision: str = "main"
) -> str:
    """
    Download a single file from a Hugging Face repository.

    Args:
        repo_id (str): The ID of the repository.
        repo_type (str): The type of the repository (e.g., "model", "dataset", "space").
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
            repo_type=repo_type,
            filename=filename,
            revision=revision,
            local_dir=output_dir,
            force_download=True,
            token=use_auth_token
        )
    except Exception as e:
        if "401 Client Error" in str(e) and "Cannot access gated repo" in str(e):
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
        metadata_file: Optional[str] = None,
        use_auth_token: Optional[str] = None,
        revision: str = "main",
        force: bool = False,
        ignore_patterns: Optional[List[str]] = None,
):
    """
    Download all files from a Hugging Face repository.

    Args:
        repo_id (str): The ID of the repository.
        repo_type (str): The type of the repository (e.g., "models", "datasets", "spaces").
        output_dir (Optional[str]): The directory to save the downloaded files.
        metadata_file (Optional[str]): The file to store and load metadata from.
        use_auth_token (Optional[str]): The authentication token to use.
        revision (str): The specific revision to download from (default: "main").
        force (bool): If True, force download all files regardless of metadata.
        ignore_patterns (Optional[List[str]]): A list of glob patterns for files to ignore.
    """
    stored_metadata = load_stored_metadata(metadata_file)
    current_metadata = get_repo_tree_metadata(repo_id, repo_type, revision, use_auth_token)

    if current_metadata is None:
        logger.error("Failed to fetch repository metadata. Aborting download.")
        return

    files_to_download = [
        file for file, metadata in current_metadata.items()
        if force or should_download_file(file, current_metadata, stored_metadata, output_dir)
    ]

    if not files_to_download:
        logger.info("All files are up to date. No download necessary.")
        return

    for file in tqdm(files_to_download, desc=f"Downloading {repo_type} files (revision: {revision})", unit="file"):
        if ignore_patterns and should_ignore_file(file, ignore_patterns):
            logger.info(f"Skipping ignored file: {file}")
            continue
        try:
            local_file = download_file(repo_id, repo_type, file, output_dir, use_auth_token, revision)
            logger.info(f"Downloaded: {local_file}")
        except Exception as e:
            logger.error(f"Error downloading {file}: {str(e)}")
            logger.info("Retrying download...")
            try:
                local_file = download_file(repo_id, repo_type, file, output_dir, use_auth_token, revision)
                logger.info(f"Successfully downloaded on retry: {local_file}")
            except Exception as e:
                logger.error(f"Failed to download {file} after retry: {str(e)}")

    # Store the new metadata for future comparisons
    store_metadata(metadata_file, current_metadata)
    logger.info(f"Download completed. Files are stored in: {output_dir}")


def process_single_repository(
        url: str,
        args: argparse.Namespace,
        use_auth_token: Optional[str],
        ignore_patterns: List[str]
):
    """
    Process a single URL or ID for repository download.

    Args:
        url (str): The URL or ID of the Hugging Face repository.
        args: The parsed command-line arguments.
        use_auth_token (Optional[str]): The authentication token to use.
        ignore_patterns (List[str]): A list of glob patterns for files to ignore.

    Returns:
        bool: True if the download was successful, False otherwise.
    """
    try:
        # Extract directory format, infer repository type, and get revision
        repo_id, formatted_repo_id, repo_type, revision = process_repo_info(url, args.revision, use_auth_token)

        # Use script directory as default if no output directory is specified
        output_dir = set_output_directory(args.output, formatted_repo_id)

        logger.info(f"Downloading repository: {repo_id}")
        logger.info(f"Revision: {args.revision}")

        # Generate the metadata file path
        metadata_file = get_metadata_file(output_dir, formatted_repo_id)

        # Main download function
        download_repo(
            repo_id,
            repo_type,
            output_dir,
            metadata_file,
            use_auth_token,
            args.revision,
            args.force,
            ignore_patterns
        )
        return True
    except KeyboardInterrupt:
        logger.warning(f"Download interrupted by user for {url}.")
        return False
    except Exception as e:
        logger.error(f"An error occurred while processing {url}: {str(e)}")
        return False


def print_summary_and_exit(
        success: bool,
        results: Dict[str, List[str]],
        execution_time: float
):
    """
    Print a summary of the download results and execution time, then exit the script.

    Args:
        success (bool): Overall success status of the downloads.
        results (Dict[str, List[str]]): Dictionary containing lists of successful and failed downloads.
        execution_time (float): Total execution time of the script.
    """
    total_downloads = len(results['success']) + len(results['failure'])
    if not total_downloads:
        sys.exit(1)

    print("\nDownload Summary")
    print("-" * 16)

    print(f"Total repositories: {total_downloads}")
    print(f"Successful: {len(results['success'])}")
    print(f"Failed: {len(results['failure'])}")
    print(f"Execution time: {execution_time:.2f} seconds")

    if results['success']:
        print("\nSuccessful downloads:")
        for url in results['success']:
            print(f"  ✓ {url}")

    if results['failure']:
        print("\nFailed downloads:")
        for url in results['failure']:
            print(f"  ✗ {url}")

    if success:
        print("\nAll downloads completed successfully.")
    elif total_downloads == 1:
        print("\nThe download failed. Please check the error messages above for details.")
    else:
        print("\nSome downloads failed. Please check the error messages above for details.")

    sys.exit(0 if success else 1)


def main():
    """
    Main function to handle command-line arguments and orchestrate the download process.
    """
    parser = argparse.ArgumentParser(description="Download files from Hugging Face.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # Command-line Arguments

    # Use a positional argument for URL, but make it optional
    parser.add_argument("url",
                        nargs='?',
                        help="URL or ID of the Hugging Face repository")
    parser.add_argument("--url-file",
                        help="File containing URLs of Hugging Face repositories")

    # Other arguments
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
    parser.add_argument("--force",
                        action="store_true",
                        help="Force re-download of all files, ignoring existing metadata and overwriting local files")

    args = parser.parse_args()

    if args.auth_help:
        print_auth_instructions()
        sys.exit(0)

    # Initialize `HfApi` and import `hf_hub_download` based on whether fast transfer mode is used
    initialize_huggingface_hub(args.fast)

    # Validate and process URLs
    urls, results = validate_and_process_urls(args, parser)
    if not urls:
        return False, results

    use_auth_token = HfFolder.get_token() if args.use_auth_token else None

    ignore_patterns = read_ignore_file(args.ignore_file)
    if ignore_patterns:
        logger.info(f"Using ignore patterns from {args.ignore_file}")

    for url in urls:
        download_success = process_single_repository(url, args, use_auth_token, ignore_patterns)
        results['success' if download_success else 'failure'].append(url)

    return len(results['failure']) == 0, results


if __name__ == "__main__":
    start_time = time.time()
    success_main, results_main = main()
    execution_time_main = time.time() - start_time

    print_summary_and_exit(success_main, results_main, execution_time_main)
