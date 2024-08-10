# Hugging Face Repository Downloader

This Python script allows you to download repositories from Hugging Face, including support for fast transfer mode when available. It can resume interrupted downloads and skip already downloaded files.

## Features

- Download entire repositories from Hugging Face
- Support for private repositories with authentication
- Resume interrupted downloads
- Skip already downloaded files
- Optional fast transfer mode for high-bandwidth connections

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
- `--auth-help`: Show instructions for setting up authentication
- `--help`: Show help message and exit

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

## Fast Transfer Mode

The `--fast` flag enables fast transfer mode, which can significantly increase download speeds on high-bandwidth connections. However, it's important to note:

1. It requires the `hf_transfer` package to be installed.
2. It lacks progress bars, making it less user-friendly for monitoring download progress.
3. It's primarily a power user tool, designed for scenarios where standard Python downloads can't fully utilize available bandwidth (typically beyond ~500MB/s).
4. It is not applicable to most users, even at most higher-speed internet connections.

By default, fast transfer mode is disabled to ensure a more user-friendly experience with progress bars and compatibility across different network conditions.

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