"""Prepare a deduplicated UTF-8 corpus with explicit source-license checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare_corpus(input_dir: Path, sources_path: Path, output_path: Path,
                   manifest_output: Path | None = None) -> dict:
    input_dir = input_dir.resolve()
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    if not isinstance(sources, dict):
        raise ValueError("Source manifest must be a JSON object keyed by relative file path")
    files = sorted(path for path in input_dir.rglob("*.txt") if path.is_file() and not path.is_symlink())
    if not files:
        raise ValueError(f"No .txt files found in {input_dir}")

    manifest_path = manifest_output or output_path.with_name(output_path.name + ".manifest.json")
    for destination in (output_path.resolve(), manifest_path.resolve()):
        if destination == input_dir or input_dir in destination.parents:
            raise ValueError("Output files must be outside the input directory")

    seen: set[str] = set()
    corpus_parts: list[str] = []
    source_records = []
    total_duplicates = 0

    for path in files:
        relative = path.relative_to(input_dir).as_posix()
        source = sources.get(relative)
        if not isinstance(source, dict):
            raise ValueError(f"Missing provenance entry for {relative}")
        license_name = source.get("license")
        source_url = source.get("source_url")
        if not isinstance(license_name, str) or not license_name.strip():
            raise ValueError(f"Missing license for {relative}")
        if not isinstance(source_url, str) or not source_url.strip():
            raise ValueError(f"Missing source_url for {relative}")
        if source.get("training_allowed") is not True:
            raise ValueError(f"Training permission is not explicitly confirmed for {relative}")

        raw = path.read_bytes()
        decoded = raw.decode("utf-8")
        normalized = unicodedata.normalize("NFC", decoded.replace("\r\n", "\n").replace("\r", "\n"))
        paragraphs = [re.sub(r"\s+", " ", item).strip()
                      for item in re.split(r"\n\s*\n", normalized)]
        kept = []
        duplicates = 0
        for paragraph in paragraphs:
            if not paragraph:
                continue
            if paragraph in seen:
                duplicates += 1
                continue
            seen.add(paragraph)
            kept.append(paragraph)
            corpus_parts.append(paragraph)
        total_duplicates += duplicates
        source_records.append({
            "file": relative,
            "source_url": source_url.strip(),
            "license": license_name.strip(),
            "training_allowed": True,
            "sha256": sha256(raw),
            "source_bytes": len(raw),
            "paragraphs_kept": len(kept),
            "duplicate_paragraphs_removed": duplicates,
        })

    if not corpus_parts:
        raise ValueError("No non-empty paragraphs were found in the licensed source files")
    corpus = "\n\n".join(corpus_parts).strip() + "\n"
    encoded = corpus.encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(encoded)
    manifest = {
        "format_version": 1,
        "preparation": "unicode_nfc_paragraph_deduplication",
        "corpus_file": output_path.name,
        "corpus_sha256": sha256(encoded),
        "corpus_bytes": len(encoded),
        "paragraphs": len(corpus_parts),
        "duplicate_paragraphs_removed": total_duplicates,
        "sources": source_records,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    return {"corpus": str(output_path), "manifest": str(manifest_path),
            "paragraphs": len(corpus_parts), "duplicates_removed": total_duplicates,
            "bytes": len(encoded), "sha256": sha256(encoded)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True,
                        help="JSON provenance manifest; every .txt file must have training_allowed=true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare_corpus(args.input_dir, args.sources, args.output,
                                    args.manifest_output), indent=2))


if __name__ == "__main__":
    main()
