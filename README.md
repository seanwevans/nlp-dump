# SpaCy Dump Tool

`dump.py` is a command-line tool designed to process text files using SpaCy, a leading NLP (Natural Language Processing) library. This script extracts and saves linguistic data in various formats, such as XML, HTML, plain text, and pickle, allowing for easy text analysis, visualization, and serialization.

## Features
- **Named Entity Recognition (NER)** for extracting structured information from unstructured text.
- **Multiple Output Formats**: Save processed text as XML, HTML, plain text, or pickle.
- **Batch Processing**: Use file glob patterns to process multiple files at once.
- **Customizable**: Choose the SpaCy model and entity types for extraction.
- **Logging and Error Handling**: Provides detailed logs for tracking progress and debugging errors.

## Requirements
- Python 3.6+
- SpaCy 3.x
- lxml
- numpy
- argparse

You can install the required dependencies with:
```bash
pip install spacy lxml numpy
```

## Installation
Clone or download the repository:
```bash
git clone https://github.com/username/spacy-dump-tool.git
cd spacy-dump-tool
```

## Usage
Run the script from the command line to process text files:
```bash
python dump.py [input_glob] [output_type] [options]
```

### Arguments
- `input_glob`: One or more glob patterns to specify input files (e.g., `*.txt` or `./data/*.md`).
- `output_type`: Format for the output files. Choose from `text`, `html`, `pickle`, `xml`, or `enterprise`.

### Options
- `-m, --model`: SpaCy model to load (default: `en_core_web_trf`).
- `-o, --out`: Output directory (defaults to the input directory).
- `-s, --skip`: Skip files that have already been processed.
- `--version`: Display the version of the tool.
- `--help`: Show help message and exit.

### Example
```bash
python dump.py "./data/*.txt" xml -o ./output -m en_core_web_sm
```
This command processes all `.txt` files in the `data` folder, applies the `en_core_web_sm` SpaCy model, and saves the results as XML files in the `output` folder.

## Output Formats
- **Text**: Plain text output with detailed token attributes.
- **HTML**: Visualize named entities using SpaCy's `displacy` renderer.
- **Pickle**: Serialized Python object containing token data.
- **XML**: Structured output with each token and its associated attributes.
- **Enterprise**: Generates both XML and HTML outputs.

## Logging
The script creates detailed logs (`error.log`) to track processed files, errors, and runtime information.

## Entity Types
The following entity types are extracted by default:
- `CARDINAL`, `DATE`, `EVENT`, `FAC`, `GPE`, `LANGUAGE`, `LAW`, `LOC`, `MONEY`, `NORP`, `ORDINAL`, `ORG`, `PERCENT`, `PERSON`, `PRODUCT`, `QUANTITY`, `TIME`, `WORK_OF_ART`

You can customize the entity list by modifying the `ents` list in the script.

## Extending the Tool
To add new output formats or customize behavior, modify the `saveas_*` functions in `dump.py`.

## License
This project is licensed under the MIT License.

## Author
Data Conversion Laboratory © 2023

