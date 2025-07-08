from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import spacy

import dump


def test_parse_args_defaults():
    args = dump.parse_args(["input.txt", "xml"])
    assert args.input_glob == ["input.txt"]
    assert args.output_type == "xml"
    assert args.model == "en_core_web_trf"
    assert args.out == Path(".")
    assert args.skip is False
    assert args.entities == dump.DEFAULT_ENTITY_TYPES
    assert args.dry_run is False


def test_parse_args_options():
    args = dump.parse_args([
        "in.md",
        "html",
        "--model",
        "en_core_web_sm",
        "--out",
        "outdir",
        "--skip",
        "--entities",
        "PERSON",
        "ORG",
        "--dry-run",
    ])
    assert args.input_glob == ["in.md"]
    assert args.output_type == "html"
    assert args.model == "en_core_web_sm"
    assert args.out == Path("outdir")
    assert args.skip is True
    assert args.entities == ["PERSON", "ORG"]
    assert args.dry_run is True


def test_main_process_file_called(monkeypatch, tmp_path):
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("small")
    f2.write_text("larger file")
    out_dir = tmp_path / "out"

    calls = []

    def fake_process_file(file_path, out_dir_arg, nlp, output_type, entity_types, skip_existing):
        calls.append((file_path, out_dir_arg, output_type, tuple(entity_types), skip_existing))
        return True

    monkeypatch.setattr(dump, "process_file", fake_process_file)
    monkeypatch.setattr(dump.spacy, "load", lambda m: spacy.blank("en"))
    monkeypatch.setattr(dump, "initialize_logs", lambda *a, **k: None)

    dump.main([str(tmp_path / "*.txt"), "json", "-o", str(out_dir), "-s"])

    assert len(calls) == 2
    assert calls[0][0].name == "a.txt"
    assert calls[1][0].name == "b.txt"
    for call in calls:
        assert call[1] == out_dir
        assert call[2] == "json"
        assert call[3] == tuple(dump.DEFAULT_ENTITY_TYPES)
        assert call[4] is True
