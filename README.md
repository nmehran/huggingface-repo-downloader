# Hugging Face Repository Downloader

This Python script allows you to download repositories from Hugging Face, including support for fast transfer mode when available. It can resume interrupted downloads and skip already downloaded files.

## Features

- Download entire repositories from Hugging Face
- Support for private repositories with authentication
- Resume interrupted downloads
- Skip already downloaded files
- Optional fast transfer mode for high-bandwidth connections
- Ignore specific files or patterns using glob syntax

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/nmehran/huggingface-repo-downloader.git
   cd huggingface-repo-downloader
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

```
python huggingface_downloader.py <repository_url> [options]
```

### Options

- `-o`, `--output`: Specify output directory (optional, default: script directory)
- `--use-auth-token`: Use the Hugging Face auth token for private or access-restricted repos
- `--revision`: Specific revision to download (default: main)
- `--fast`: Enable fast transfer mode (requires `hf_transfer` package)
- `--ignore-file`: Specify a file containing glob patterns to ignore
- `--force`: Force re-download of all files, ignoring existing metadata
- `--help`: Show help message and exit
- `--auth-help`: Show instructions for setting up authentication

### Examples

1. Download a public repository:
   ```
   python huggingface_downloader.py meta-llama/Meta-Llama-3.1-8B-Instruct/tree/main
   ```

2. Download a private repository with authentication:
   ```
   python huggingface_downloader.py your-private-repo/model --use-auth-token
   ```

3. Download a specific revision:
   ```
   python huggingface_downloader.py facebook/bart-large --revision v1.0
   ```

4. Use fast transfer mode:
   ```
   python huggingface_downloader.py google/t5-v1_1-large --fast
   ```

5. Use an ignore file to skip certain files or patterns:
   ```
   python huggingface_downloader.py meta-llama/Meta-Llama-3.1-8B-Instruct --ignore-file .hfignore
   ```

6. Force re-download of all files:
   ```
   python huggingface_downloader.py meta-llama/Meta-Llama-3.1-8B-Instruct --force
   ```
   
## Fast Transfer Mode

The `--fast` flag enables fast transfer mode, which can significantly increase download speeds on high-bandwidth connections. However, it's important to note:

1. It requires the `hf_transfer` package to be installed.
2. It lacks progress bars, making it less user-friendly for monitoring download progress.
3. It's primarily a power user tool, designed for scenarios where standard Python downloads can't fully utilize available bandwidth (typically beyond ~500MB/s).
4. It is not applicable to most users, even at most higher-speed internet connections.

By default, fast transfer mode is disabled to ensure a more user-friendly experience with progress bars and compatibility across different network conditions.

## Ignore File

You can use an ignore file to specify patterns of files or directories that should be skipped during the download. The ignore file uses glob syntax for pattern matching.

To use an ignore file:

1. Create a text file (e.g., `.hfignore`) with one pattern per line.
2. Use the `--ignore-file` option when running the script, pointing to your ignore file.

Example `.hfignore` file content:
```
*.txt
test_data/*
logs/
```

This would ignore all `.txt` files, everything in the `test_data` directory, and the `logs` directory.

Note:
- Lines starting with `#` (including those with leading whitespace) are treated as comments and ignored.
- Empty lines are skipped.

## Metadata Handling

The script uses metadata to keep track of downloaded files and their versions. This helps to avoid unnecessary re-downloads of unchanged files. The metadata is stored in a JSON file in the output directory.

When you run the script:
- It compares the current repository metadata with the stored metadata.
- Only files that are new or have changed since the last download are downloaded.
- The `--force` option overrides this behavior and re-downloads all files.

## Authentication

For private or access-restricted repositories, you need to set up authentication. Follow these steps:

1. Visit https://huggingface.co/settings/tokens to generate a token.
2. Run 'huggingface-cli login' and enter your token when prompted.

For detailed instructions, run:
```
python huggingface_downloader.py --auth-help
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.