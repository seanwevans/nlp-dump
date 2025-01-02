#!/usr/bin/env python

"""  dump.py - Dump SpaCy data  """


import argparse
from glob import glob
import logging
from pathlib import Path
import pickle
import os
import re
import sys
from time import process_time
import unicodedata

from lxml import etree as ET
import numpy
import spacy

PROG = Path(__file__).stem
VERSION = "1.6.0"

SPACY_CHAR_LIM = 1_000_000
logger = logging.getLogger()

ents = [
    "CARDINAL",  # numerals that do not fall under another type.
    "DATE",  # absolute or relative dates or periods.
    "EVENT",  # named events such as hurricanes, battles, wars, sports events, etc.
    "FAC",  # facilities such as buildings, airports, highways, bridges, etc.
    "GPE",  # countries, cities, states.
    "LANGUAGE",  # any named language.
    "LAW",  # named documents made into laws.
    "LOC",  # non-GPE locations such as mountain ranges, bodies of water, etc.
    "MONEY",  # monetary values, including unit.
    "NORP",  # nationalities or religious or political groups.
    "ORDINAL",  # “first”, “second”, etc.
    "ORG",  # organizations such as companies, agencies, institutions, etc.
    "PERCENT",  # percentage, including “%”.
    "PERSON",  # people, including fictitious.
    "PRODUCT",  # non-service products such as objects, vehicles, foods, etc.
    "QUANTITY",  # measurements such as weight, distance, force, etc.
    "TIME",  # times smaller than a day.
    "WORK_OF_ART",  # titles of books, songs, etc.
]


def parse_args(args):
    argp = argparse.ArgumentParser(
        prog=PROG,
        description=__doc__,        
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
    )

    argp.add_argument("input_glob", nargs="+", help="input glob(s)")

    argp.add_argument(
        "output_type",
        default="xml",
        type=str.lower,
        choices=["text", "html", "pickle", "xml", "enterprise"],
        help="Output type",
    )
    argp.add_argument(
        "-m", "--model", default="en_core_web_trf", help="name of model to load"
    )
    argp.add_argument(
        "-o",
        "--out",
        default="",
        help="Directory to output results. Defaults to input dir.",
    )
    argp.add_argument(
        "-s",
        "--skip",
        action="store_true",
        help="set flag to skip if output file exists.",
    )

    argp.add_argument(
        "--version",
        action="version",
        version=f"{PROG} v{VERSION} © Data Conversion Laboratory 2023",
    )

    argp.add_argument(
        "--help",
        action="help",
        help="show this help message and exit",
    )

    return argp.parse_args(args)


def initialize_logs(
    params=None,
    log_file=None,
    stream_level=logging.DEBUG,
    file_level=logging.DEBUG,
    log_format="%(asctime)-23s %(module)s.%(funcName)s %(levelname)-8s %(message)s",
):
    """initialize logs"""

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


def remove_control_chars(s):
    return "".join(
        ch for ch in s if unicodedata.category(ch)[0] != "C" and ch != "\x00"
    )


def saveas_text(doc, out_file):
    out = []

    for token in doc:
        if re.search(r"^\s*$", token.text):
            continue

        out.append(token.text + "\n")
        for method in dir(token):
            if method in ["doc", "vocab", "vector", "tensor", "_"] or method.startswith(
                "__"
            ):
                continue

            m = f"token.{method}"
            to_examine = eval(m)

            if str(type(to_examine)) == "<class 'builtin_function_or_method'>":
                continue

            line = []
            line.append(f"  {method:20s}")
            if (
                isinstance(to_examine, str)
                or isinstance(to_examine, int)
                or isinstance(to_examine, float)
                or isinstance(to_examine, tuple)
                or isinstance(to_examine, bool)
                or isinstance(to_examine, numpy.ndarray)
                or isinstance(to_examine, numpy.float32)
                or to_examine is None
            ):
                line.append(str(to_examine).strip())
            elif "__iter__" in dir(to_examine):
                line.append(str([str(i).strip() for i in to_examine]))
            elif isinstance(to_examine, spacy.tokens.token.Token):
                line.append(to_examine.text.strip())
            else:
                line.append(str(type(to_examine)))

            out.append(" ".join(line) + "\n")

        out.append("\n")

    with open(out_file, "w", encoding="UTF-8") as g:
        g.writelines(out)


def saveas_html(doc, out_file):
    # options = {"ents": ["PERSON", "ORG", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY"]}
    options = {
        "ents": [
            "CARDINAL",
            "DATE",
            "EVENT",
            "FAC",
            "GPE",
            "LANGUAGE",
            "LAW",
            "LOC",
            "MONEY",
            "NORP",
            "ORDINAL",
            "ORG",
            "PERCENT",
            "PERSON",
            "PRODUCT",
            "QUANTITY",
            "TIME",
            "WORK_OF_ART",
        ]
    }
    html = spacy.displacy.render(doc, style="ent", options=options, page=True)
    html = re.sub(r"(?:</br>\s*){2,}", "</br> ", html)

    with open(str(out_file), "w", encoding="UTF-8") as g:
        g.write(html)


def saveas_pickle(doc, out_file):
    spacy_dump = []

    for token in doc:
        if re.search(r"^\s*$", token.text):
            continue

        methods = {}

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
                isinstance(to_examine, str)
                or isinstance(to_examine, int)
                or isinstance(to_examine, float)
                or isinstance(to_examine, tuple)
                or isinstance(to_examine, bool)
                or isinstance(to_examine, numpy.ndarray)
                or isinstance(to_examine, numpy.float32)
                or to_examine is None
            ):
                methods[method] = str(to_examine)
            elif "__iter__" in dir(to_examine):
                methods[method] = str([str(i).strip() for i in to_examine])
            elif isinstance(to_examine, spacy.tokens.token.Token):
                methods[method] = to_examine.text.strip()
            else:
                methods[method] = str(type(to_examine))

            spacy_dump.append([token.text, methods])

    with open(out_file, "w") as g:
        pickle.dump(spacy_dump, g)


def saveas_xml(doc, out_file):
    root = ET.Element("document")
    tree = ET.ElementTree(root)

    for token in doc:
        if re.search(r"^\s*$", token.text):
            continue

        ttext = ET.SubElement(root, "word")
        ttext.set("value", remove_control_chars(token.text.strip()))

        for method in dir(token):
            if method in ["doc", "vocab", "vector", "tensor", "_"] or method.startswith(
                "__"
            ):
                continue

            m = f"token.{method}"
            to_examine = eval(m)

            if str(type(to_examine)) == "<class 'builtin_function_or_method'>":
                continue

            meth = ET.SubElement(ttext, "method")
            meth.set("value", method)
            if (
                isinstance(to_examine, str)
                or isinstance(to_examine, int)
                or isinstance(to_examine, float)
                or isinstance(to_examine, tuple)
                or isinstance(to_examine, bool)
                or isinstance(to_examine, numpy.ndarray)
                or isinstance(to_examine, numpy.float32)
                or to_examine is None
            ):
                meth.text = remove_control_chars(str(to_examine).strip())
            elif "__iter__" in dir(to_examine):
                meth.text = str([str(i).strip() for i in to_examine])
            elif isinstance(to_examine, spacy.tokens.token.Token):
                meth.text = remove_control_chars(to_examine.text.strip())
            else:
                meth.text = str(type(to_examine))

    tree.write(str(out_file), encoding="UTF-8", pretty_print=True)


def saveas(out_type, doc, out_file):
    if out_type == "text":
        saveas_txt(doc, out_file)

    if out_type == "html":
        saveas_html(doc, out_file)

    if out_type == "pickle":
        saveas_pickle(doc, out_file)

    if out_type == "xml":
        saveas_xml(doc, out_file)

    if out_type == "enterprise":
        saveas_xml(doc, out_file.with_suffix(".xml"))
        saveas_html(doc, out_file.with_suffix(".html"))


def main(args):
    params = parse_args(args)
    initialize_logs(params, "error.log")

    logger.info("compiling file list...")
    files_to_process = []
    for ig in params.input_globs:
        files_to_process.extend(glob(ig, recursive=True))

    if len(files_to_process) == 0:
        logger.info("no files found, exiting...")
        sys.exit()
    else:
        logger.info("%d files to process", len(files_to_process))

    logger.info("Loading text model...")
    nlp = spacy.load(params.model)
    logger.info("model loaded")

    files_to_process.sort(key=os.path.getsize)
    doc_count = 0

    for in_file in files_to_process:
        in_file = Path(in_file)
        out_file = in_file.with_suffix(f".{params.output_type}")

        if params.skip and out_file.exists():
            continue

        logger.info("%d: %s", doc_count + 1, in_file.resolve())
        doc_count += 1

        try:
            with open(in_file, "r", encoding="ansi") as f:
                doc_text = f.read()
        except Exception as e:
            logger.error(e)
            continue

        try:
            doc = nlp(doc_text[:SPACY_CHAR_LIM])
        except RuntimeError as err:
            logger.error(f"Failed nlp: {err}, skipping...")
            continue

        saveas(format=params.output_type, doc=doc, out_file=out_file)

        if Path(out_file).exists():
            logger.info(out_file.resolve())
        else:
            logger.info("Failed to produce %s", out_file)


if __name__ == "__main__":
    main(sys.argv[1:])
