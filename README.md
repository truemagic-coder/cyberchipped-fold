# CyberChipped-Fold

CyberChipped-Fold is a streamlined version of ColabFold that focuses on running AlphaFold2 locally on Linux systems. It combines the power of ColabFold and LocalColabFold to provide an easy-to-use tool for protein structure prediction.

## Features

- Supports AlphaFold2 for protein structure prediction
- Includes `colabfold_search` and `colabfold_batch` functionalities
- Optimized for Linux systems

## Installation

1. Clone this repository or download it to your local machine.
2. Navigate to the cyberchipped-fold directory:
   ```
   cd /path/to/cyberchipped-fold
   ```
3. Run the installation script:
   ```
   bash install.sh
   ```
4. After installation, add the cyberchipped-fold conda environment to your PATH:
   ```
   export PATH="/path/to/cyberchipped-fold/cyberchipped-fold-conda/bin:$PATH"
   ```

## Usage

### colabfold_search

To use `colabfold_search`:

```
colabfold_search [OPTIONS] QUERY_FILE RESULTS_DIR
```

For more details on available options, run:

```
colabfold_search --help
```

### colabfold_batch

To use `colabfold_batch`:

```
colabfold_batch [OPTIONS] INPUT_DIR OUTPUT_DIR
```

For more details on available options, run:

```
colabfold_batch --help
```

## Support

For issues and feature requests, please open an issue on the GitHub repository.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

CyberChipped-FOld is based on the work of ColabFold and LocalColabFold. We thank the original authors and contributors of these projects for their valuable work in the field of protein structure prediction.