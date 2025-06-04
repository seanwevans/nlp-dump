# SpaCy Dump Tool

`dump.py` is a powerful command-line tool designed to process text files using 
SpaCy, a leading Natural Language Processing library. This script extracts and
saves linguistic data in various formats, such as XML, HTML, JSON, plain text,
and pickle, allowing for easy text analysis, visualization, and serialization.

## Features

- **Named Entity Recognition (NER)** for extracting structured information from unstructured text
- **Multiple Output Formats**: Save processed text as XML, HTML, JSON, plain text, or pickle
- **Batch Processing**: Use file glob patterns to process multiple files at once
- **Customizable**: Choose the SpaCy model and specific entity types for extraction
- **Comprehensive Logging**: Detailed logs for tracking progress and debugging errors
- **Enhanced Visualization**: Color-coded entity highlighting in HTML output
- **Robust Error Handling**: Graceful handling of encoding issues and large files

## Requirements

- Python 3.6+
- SpaCy 3.x
- lxml
- numpy

You can install the required dependencies with:

```bash
pip install spacy lxml numpy
```

You'll also need to download at least one SpaCy model:

```bash
python -m spacy download en_core_web_sm  # Smaller, faster model
# or
python -m spacy download en_core_web_trf  # Larger, more accurate model (default)
```

## Installation

Clone or download the repository:

```bash
git clone https://github.com/seanwevans/nlp-dump.git
cd nlp-dump
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the script from the command line to process text files:

```bash
python dump.py [options] [input_glob(s)] [output_type]
```

### Arguments

- `input_glob`: One or more glob patterns to specify input files (e.g., `*.txt` or `./data/*.md`)
- `output_type`: Format for the output files. Choose from `text`, `html`, `json`, `pickle`, `xml`, or `enterprise` (default: `xml`)

### Options

- `-m, --model`: SpaCy model to load (default: `en_core_web_trf`)
- `-o, --out`: Output directory (default: current directory)
- `-s, --skip`: Skip files which have already been processed (i.e. their specified output already exists)
- `-e, --entities`: Specify which entity types to include (space-separated list)
- `-d, --dry-run`: Show files to be processed without actually processing them
- `--version`: Display the version information
- `--help`: Show help message and exit

### Examples

Basic usage with default settings:
```bash
python dump.py "./data/*.txt" xml
```

Process markdown files with a specific model and save to a custom directory:
```bash
python dump.py "./docs/**/*.md" html -o ./output -m en_core_web_sm
```

Process multiple file types, skipping existing outputs:
```bash
python dump.py "./data/*.txt" "./papers/*.md" json -s
```

Only extract specific entity types:
```bash
python dump.py "./news/*.txt" html -e PERSON ORG GPE DATE
```

Preview which files would be processed:
```bash
python dump.py "./data/**/*.txt" xml -d
```

## Output Formats

### Text
Plain text output with detailed token attributes for each token in the document:
```
Token
  attribute1          value1
  attribute2          value2
  ...
```

### HTML
Visualize named entities using SpaCy's `displacy` renderer with color-coded highlights for different entity types.

### JSON
Structured output containing token data and entities in a JSON format:
```json
{
  "metadata": {
    "spacy_model": "en",
    "tokens": 42,
    "sentences": 3
  },
  "tokens": [
    {
      "text": "Apple",
      "lemma": "apple",
      "pos": "PROPN",
      "tag": "NNP",
      "dep": "nsubj",
      "is_stop": false,
      "is_alpha": true,
      "is_punct": false
    },
    ...
  ],
  "entities": [
    {
      "text": "Apple",
      "start_char": 0,
      "end_char": 5,
      "label": "ORG"
    },
    ...
  ]
}
```

### XML
Structured output with each token and its associated attributes:
```xml
<document>
  <word value="Apple">
    <method value="lemma_">apple</method>
    <method value="pos_">PROPN</method>
    ...
  </word>
  ...
</document>
```

### Pickle
Serialized Python object containing token data for programmatic use.

### Enterprise
Generates XML, HTML, and JSON outputs simultaneously.

## Logging

The script creates detailed logs (`spacy.dump`) to track processed files, errors, and runtime information.

## Entity Types

The following entity types are extracted by default:

| Type | Description |
|------|-------------|
| CARDINAL | Numerals that do not fall under another type |
| DATE | Absolute or relative dates or periods |
| EVENT | Named events such as hurricanes, battles, sports events |
| FAC | Facilities such as buildings, airports, bridges |
| GPE | Countries, cities, states |
| LANGUAGE | Any named language |
| LAW | Named documents made into laws |
| LOC | Non-GPE locations such as mountain ranges, bodies of water |
| MONEY | Monetary values, including unit |
| NORP | Nationalities or religious or political groups |
| ORDINAL | "first", "second", etc. |
| ORG | Organizations such as companies, agencies, institutions |
| PERCENT | Percentage, including "%" |
| PERSON | People, including fictitious |
| PRODUCT | Objects, vehicles, foods, etc. |
| QUANTITY | Measurements such as weight, distance |
| TIME | Times smaller than a day |
| WORK_OF_ART | Titles of books, songs, etc. |

You can customize which entity types to include using the `-e` option.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.