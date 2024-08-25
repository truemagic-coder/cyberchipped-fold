# Folding

[![PyPI - Version](https://img.shields.io/pypi/v/folding)](https://pypi.org/project/folding/)

Folding is a streamlined version of ColabFold that focuses on running AlphaFold2 locally on Linux systems. It combines the power of ColabFold and LocalColabFold to provide an easy-to-use tool for protein structure prediction.

## Features

- Supports AlphaFold2 for protein structure prediction
- Includes `folding-search` and `folding-batch` functionalities

## Requirements
* Linux
* RTX 4070 or greater

## Installation

1. Clone this repository or download it to your local machine.
2. Navigate to the folding directory:
   ```
   cd /path/to/folding
   ```
3. Run the installation script:
   ```
   bash install.sh
   ```
4. After installation, add the folding conda environment to your PATH:
   ```
   export PATH="/path/to/folding/folding-conda/bin:$PATH"
   ```

## Usage

### folding-search

To use `folding-search`:

```
folding-search [OPTIONS] QUERY_FILE RESULTS_DIR
```

For more details on available options, run:

```
folding-search --help
```

### folding-batch

To use `folding-batch`:

```
folding-batch [OPTIONS] INPUT_DIR OUTPUT_DIR
```

For more details on available options, run:

```
folding-batch --help
```

## Support

For issues and feature requests, please open an issue on the GitHub repository.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

folding is based on the work of ColabFold and LocalColabFold. We thank the original authors and contributors of these projects for their valuable work in the field of protein structure prediction.