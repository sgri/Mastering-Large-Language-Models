#!/usr/bin/env bash
set -e

# Function to print usage
usage() {
    cat <<EOF
Usage: $0 [-h] <folder>
Convert all .ipynb files in the specified folder (including subfolders) to .py files.

Options:
    -h  Display this help message
EOF
}

# Check if jupyter nbconvert is available
if ! command -v jupyter &> /dev/null; then
    echo "Error: jupyter nbconvert is not available. Please install Jupyter."
    exit 1
fi

# Parse command-line options
while getopts ":h" opt; do
    case ${opt} in
        h )
            usage
            exit 0
            ;;
        \? )
            echo "Invalid option: -$OPTARG" >&2
            usage
            exit 1
            ;;
    esac
done
shift $((OPTIND -1))

# Check if the folder argument is provided
if [[ -z "$1" ]]; then
    echo "Error: No folder specified."
    usage
    exit 1
fi

# Folder to search for .ipynb files
FOLDER="$1"

# Check if the folder exists
if [[ ! -d "$FOLDER" ]]; then
    echo "Error: Specified folder does not exist."
    exit 1
fi

# Convert each .ipynb file to .py
find "$FOLDER" -name '*.ipynb' |
while read -r file; do
    jupyter nbconvert --to script "$file"
    python_file="${file%.*}.py"
    if [ -f "$python_file" ]; then
      # Fix a typo in Chapter 03
      sed -i "s/from spellchecker import SpellChecker/from spellchecker import Spellchecker as SpellChecker/" "$python_file"
      sed -i "1a\# Generated from '$(basename "$file")' with $(basename $0)." "$python_file"
    fi
done