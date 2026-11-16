import json
import tempfile
import unittest
from pathlib import Path

from core.prepare_corpus import prepare_corpus


class CorpusPreparationTests(unittest.TestCase):
    def test_normalizes_deduplicates_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "raw"
            source_dir.mkdir()
            (source_dir / "a.txt").write_text("cafe\u0301\n\nShared paragraph.\n", encoding="utf-8")
            (source_dir / "b.txt").write_text("Shared paragraph.\n\nSecond source paragraph.\n", encoding="utf-8")
            sources = {
                "a.txt": {"source_url": "https://example.org/a", "license": "CC0-1.0", "training_allowed": True},
                "b.txt": {"source_url": "https://example.org/b", "license": "CC0-1.0", "training_allowed": True},
            }
            source_file = root / "sources.json"
            source_file.write_text(json.dumps(sources), encoding="utf-8")
            output = root / "prepared" / "corpus.txt"
            stats = prepare_corpus(source_dir, source_file, output)
            text = output.read_text(encoding="utf-8")
            manifest = json.loads(Path(stats["manifest"]).read_text(encoding="utf-8"))
            self.assertIn("café", text)
            self.assertEqual(text.count("Shared paragraph."), 1)
            self.assertEqual(stats["duplicates_removed"], 1)
            self.assertEqual(manifest["sources"][0]["license"], "CC0-1.0")
            self.assertEqual(manifest["corpus_sha256"], stats["sha256"])

    def test_requires_explicit_training_permission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "raw"
            source_dir.mkdir()
            (source_dir / "text.txt").write_text("sample text", encoding="utf-8")
            source_file = root / "sources.json"
            source_file.write_text(json.dumps({
                "text.txt": {"source_url": "https://example.org", "license": "unknown",
                             "training_allowed": False}
            }), encoding="utf-8")
            with self.assertRaises(ValueError):
                prepare_corpus(source_dir, source_file, root / "corpus.txt")

    def test_requires_provenance_for_every_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "raw"
            source_dir.mkdir()
            (source_dir / "unlisted.txt").write_text("sample text", encoding="utf-8")
            source_file = root / "sources.json"
            source_file.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                prepare_corpus(source_dir, source_file, root / "corpus.txt")


if __name__ == "__main__":
    unittest.main()
