#!/usr/bin/env python

"""
dump.py - Process and analyze text files using SpaCy NLP

This script extracts linguistic data from text files using SpaCy and saves
the results in various formats (XML, HTML, JSON, text, pickle).
"""

import argparse
import json
import logging
from glob import glob
from pathlib import Path
import pickle
import re
import sys
from time import process_time
import unicodedata
from typing import List, Dict, Optional

from lxml import etree as ET
import numpy
import spacy
from spacy import displacy

PROG = Path(__file__).stem
VERSION = "1.7.0"

SPACY_CHAR_LIM = 1_000_000
logger = logging.getLogger(__name__)

DEFAULT_ENTITY_TYPES = [
    "CARDINAL",  # numerals that do not fall under another type
    "DATE",  # absolute or relative dates or periods
    "EVENT",  # named events such as hurricanes, battles, wars, sports events, etc.
    "FAC",  # facilities such as buildings, airports, highways, bridges, etc.
    "GPE",  # countries, cities, states
    "LANGUAGE",  # any named language
    "LAW",  # named documents made into laws
    "LOC",  # non-GPE locations such as mountain ranges, bodies of water, etc.
    "MONEY",  # monetary values, including unit
    "NORP",  # nationalities or religious or political groups
    "ORDINAL",  # "first", "second", etc.
    "ORG",  # organizations such as companies, agencies, institutions, etc.
    "PERCENT",  # percentage, including "%"
    "PERSON",  # people, including fictitious
    "PRODUCT",  # objects, vehicles, foods, etc.
    "QUANTITY",  # measurements such as weight, distance, etc.
    "TIME",  # times smaller than a day
    "WORK_OF_ART",  # titles of books, songs, etc.
]


def parse_args(args: List[str]) -> argparse.Namespace:
    """
    Parse command line arguments.

    Args:
        args: Command line arguments

    Returns:
        Parsed arguments
    """
    argp = argparse.ArgumentParser(
        prog=PROG,
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    argp.add_argument("input_glob", nargs="+", help="Input glob pattern(s)")

    argp.add_argument(
        "output_type",
        type=str.lower,
        choices=["text", "html", "pickle", "xml", "json", "enterprise"],
        default="xml",
        help="Output format type",
    )

    argp.add_argument(
        "-m",
        "--model",
        type=str,
        default="en_core_web_trf",
        help="Name of SpaCy model to load",
    )

    argp.add_argument(
        "-o",
        "--out",
        type=Path,
        default=".",
        help="Directory to output results",
    )

    argp.add_argument(
        "-s",
        "--skip",
        action="store_true",
        help="Skip processing if output file already exists",
    )

    argp.add_argument(
        "-e",
        "--entities",
        nargs="+",
        default=DEFAULT_ENTITY_TYPES,
        help="Entity types to include (space-separated list)",
    )

    argp.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        help="Show files to be processed without actually processing them",
    )

    argp.add_argument(
        "--version",
        action="version",
        version=f"{PROG} v{VERSION}",
    )

    return argp.parse_args(args)


def initialize_logs(
    params: Optional[argparse.Namespace] = None,
    log_file: Optional[str] = None,
    stream_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_format: str = "%(asctime)-23s %(module)s.%(funcName)s %(levelname)-8s %(message)s",
) -> None:
    """
    Initialize logging system.

    Args:
        params: Command line parameters to log
        log_file: Path to log file
        stream_level: Log level for console output
        file_level: Log level for file output
        log_format: Format string for log messages
    """
    logger.setLevel(logging.DEBUG)

    for handler in logger.handlers:
        logger.removeHandler(handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(log_format))
    stream_handler.setLevel(stream_level)
    logger.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="UTF-8")
        file_handler.setFormatter(logging.Formatter(log_format))
        file_handler.setLevel(file_level)
        logger.addHandler(file_handler)

    logger.info("🍍 Invocation: %s v%s", PROG, VERSION)

    if params:
        logger.debug("Parameters:")
        for param, value in vars(params).items():
            logger.debug("  %-16s%s", param, value)

    logger.debug("Logs:")
    for log_handle in logger.handlers:
        logger.debug("  %s", log_handle)


def remove_control_chars(s: str) -> str:
    """
    Remove control characters from a string.

    Args:
        s: Input string

    Returns:
        String with control characters removed
    """
    return "".join(
        ch for ch in s if unicodedata.category(ch)[0] != "C" and ch != "\x00"
    )


def get_token_attributes(token: spacy.tokens.token.Token) -> Dict[str, str]:
    """
    Extract all available attributes from a SpaCy token.

    Args:
        token: SpaCy token

    Returns:
        Dictionary of attribute name to string value
    """
    attributes = {}

    for method in dir(token):
        if method in ["doc", "vocab", "vector", "tensor", "_"] or method.startswith(
            "__"
        ):
            continue

        m = f"token.{method}"
        to_examine = eval(m)

        if str(type(to_examine)) == "<class 'builtin_function_or_method'>":
            continue

        if (
            isinstance(
                to_examine, (str, int, float, tuple, bool, numpy.ndarray, numpy.float32)
            )
            or to_examine is None
        ):
            attributes[method] = str(to_examine).strip()
        elif hasattr(to_examine, "__iter__"):
            attributes[method] = str([str(i).strip() for i in to_examine])
        elif isinstance(to_examine, spacy.tokens.token.Token):
            attributes[method] = to_examine.text.strip()
        else:
            attributes[method] = str(type(to_examine))

    return attributes


def saveas_text(doc: spacy.tokens.doc.Doc, out_file: Path) -> None:
    """
    Save SpaCy doc as text file with detailed token information.

    Args:
        doc: SpaCy document
        out_file: Output file path
    """
    out = []

    for token in doc:
        if re.search(r"^\s*$", token.text):
            continue

        out.append(token.text + "\n")

        attributes = get_token_attributes(token)
        for attr_name, attr_value in attributes.items():
            line = [f"  {attr_name:20s}", attr_value]
            out.append(" ".join(line) + "\n")

        out.append("\n")

    with open(out_file, "w", encoding="UTF-8") as g:
        g.writelines(out)


def saveas_html(
    doc: spacy.tokens.doc.Doc,
    out_file: Path,
    entity_types: List[str] = DEFAULT_ENTITY_TYPES,
) -> None:
    """
    Save SpaCy doc as HTML visualization using displacy.

    Args:
        doc: SpaCy document
        out_file: Output file path
        entity_types: Entity types to highlight
    """
    options = {"ents": entity_types}

    colors = {}
    for ent in entity_types:
        ent_hash = sum(ord(c) for c in ent) % 360
        colors[ent] = f"hsl({ent_hash}, 70%, 50%)"

    options["colors"] = colors

    html = displacy.render(doc, style="ent", options=options, page=True)

    html = re.sub(r"(?:</br>\s*){2,}", "</br> ", html)

    with open(str(out_file), "w", encoding="UTF-8") as g:
        g.write(html)


def saveas_pickle(doc: spacy.tokens.doc.Doc, out_file: Path) -> None:
    """
    Save SpaCy doc as pickle file.

    Args:
        doc: SpaCy document
        out_file: Output file path
    """
    spacy_dump = []

    for token in doc:
        if re.search(r"^\s*$", token.text):
            continue

        attributes = get_token_attributes(token)
        spacy_dump.append([token.text, attributes])

    with open(out_file, "wb") as g:  # Use binary mode for pickle
        pickle.dump(spacy_dump, g)


def saveas_xml(doc: spacy.tokens.doc.Doc, out_file: Path) -> None:
    """
    Save SpaCy doc as XML file.

    Args:
        doc: SpaCy document
        out_file: Output file path
    """
    root = ET.Element("document")
    tree = ET.ElementTree(root)

    for token in doc:
        if re.search(r"^\s*$", token.text):
            continue

        ttext = ET.SubElement(root, "word")
        ttext.set("value", remove_control_chars(token.text.strip()))

        attributes = get_token_attributes(token)
        for attr_name, attr_value in attributes.items():
            meth = ET.SubElement(ttext, "method")
            meth.set("value", attr_name)
            meth.text = remove_control_chars(attr_value)

    tree.write(str(out_file), encoding="UTF-8", pretty_print=True)


def saveas_json(doc: spacy.tokens.doc.Doc, out_file: Path) -> None:
    """
    Save SpaCy doc as JSON file.

    Args:
        doc: SpaCy document
        out_file: Output file path
    """
    result = {
        "metadata": {
            "spacy_model": doc.vocab.lang,
            "tokens": len(doc),
            "sentences": len(list(doc.sents)),
        },
        "tokens": [],
        "entities": [],
    }

    for token in doc:
        if re.search(r"^\s*$", token.text):
            continue

        token_data = {
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "tag": token.tag_,
            "dep": token.dep_,
            "is_stop": token.is_stop,
            "is_alpha": token.is_alpha,
            "is_punct": token.is_punct,
        }
        result["tokens"].append(token_data)

    for ent in doc.ents:
        entity_data = {
            "text": ent.text,
            "start_char": ent.start_char,
            "end_char": ent.end_char,
            "label": ent.label_,
        }
        result["entities"].append(entity_data)

    with open(out_file, "w", encoding="UTF-8") as g:
        json.dump(result, g, indent=2, ensure_ascii=False)


def saveas(
    output_type: str,
    doc: spacy.tokens.doc.Doc,
    out_file: Path,
    entity_types: List[str] = DEFAULT_ENTITY_TYPES,
) -> None:
    """
    Save SpaCy doc in the specified format.

    Args:
        output_type: Output format (text, html, pickle, xml, json, enterprise)
        doc: SpaCy document
        out_file: Output file path
        entity_types: Entity types to include in output
    """
    if output_type == "text":
        saveas_text(doc, out_file)
    elif output_type == "html":
        saveas_html(doc, out_file, entity_types)
    elif output_type == "pickle":
        saveas_pickle(doc, out_file)
    elif output_type == "xml":
        saveas_xml(doc, out_file)
    elif output_type == "json":
        saveas_json(doc, out_file)
    elif output_type == "enterprise":
        saveas_xml(doc, out_file.with_suffix(".xml"))
        saveas_html(doc, out_file.with_suffix(".html"), entity_types)
        saveas_json(doc, out_file.with_suffix(".json"), entity_types)


def process_file(
    file_path: Path,
    nlp: spacy.language.Language,
    output_type: str,
    entity_types: List[str],
    skip_existing: bool = False,
) -> bool:
    """
    Process a single file with SpaCy.

    Args:
        file_path: Path to input file
        nlp: Loaded SpaCy model
        output_type: Output format
        entity_types: Entity types to include
        skip_existing: Skip if output file exists

    Returns:
        Success status
    """
    out_file = file_path.with_suffix(f".{output_type}")

    if skip_existing and out_file.exists():
        logger.debug("Skipping existing file: %s", out_file)
        return True

    try:
        try:
            with open(file_path, "r", encoding="UTF-8") as f:
                doc_text = f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, "r", encoding="ansi") as f:
                    doc_text = f.read()
            except UnicodeDecodeError:
                with open(file_path, "r", encoding="latin-1") as f:
                    doc_text = f.read()

        doc = nlp(doc_text[:SPACY_CHAR_LIM])
        saveas(
            output_type=output_type,
            doc=doc,
            out_file=out_file,
            entity_types=entity_types,
        )
        logger.debug("Successfully processed: %s", out_file)

    except Exception as e:
        logger.error("Error processing %s: %s", file_path, str(e))
        return False

    return True


def main(args: List[str]) -> None:
    """
    Main function to process files with SpaCy.

    Args:
        args: Command line arguments
    """

    params = parse_args(args)
    initialize_logs(params, "spacy.dump")

    logger.info("Compiling file list...")
    files_to_process = []
    for input_glob in params.input_glob:
        files_to_process.extend(glob(input_glob, recursive=True))

    if not files_to_process:
        logger.info("No files found, exiting...")
        sys.exit(0)
    else:
        logger.info("Found %d files to process", len(files_to_process))

    files_to_process = [Path(f) for f in files_to_process]
    files_to_process.sort(key=lambda p: p.stat().st_size)

    if params.dry_run:
        logger.info("Dry run - would process these files:")
        for i, file_path in enumerate(files_to_process, 1):
            logger.info("  %d: %s", i, file_path.resolve())
        sys.exit(0)

    logger.info("Loading model: %s...", params.model)
    start_time = process_time()
    try:
        nlp = spacy.load(params.model)
        logger.info("Model loaded in %.2f seconds", process_time() - start_time)
    except OSError as e:
        logger.error("Failed to load model: %s", str(e))
        logger.info("Download it first with: python -m spacy download %s", params.model)
        sys.exit(1)

    processed_count = 0
    success_count = 0

    params.out.mkdir(parents=True, exist_ok=True)

    logger.info("Processing files...")
    for i, file_path in enumerate(files_to_process, start=1):
        processed_count += 1

        logger.info(
            "Processing file %d/%d: %s", i, len(files_to_process), file_path.name
        )

        if process_file(
            file_path, nlp, params.output_type, params.entities, params.skip
        ):
            success_count += 1
            logger.info("Successfully processed %s", file_path.name)

    logger.info(
        "Processing complete: %d/%d files successfully processed",
        success_count,
        processed_count,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
