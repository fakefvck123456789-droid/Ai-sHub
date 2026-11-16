import tempfile
import unittest
from pathlib import Path
from core.validate_bilingual_corpus import inspect_bilingual_corpus

class BilingualCorpusTests(unittest.TestCase):
    def test_accepts_bangla_and_english(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.txt"
            path.write_text(("বাংলা ভাষা AiHub language model. " * 20), encoding="utf-8")
            result = inspect_bilingual_corpus(path)
            self.assertTrue(result["balanced_enough"])
            self.assertGreater(result["bangla_share"], 0)
            self.assertGreater(result["english_share"], 0)

    def test_rejects_english_only_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "english.txt"
            path.write_text("English only language model. " * 30, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Bilingual corpus check failed"):
                inspect_bilingual_corpus(path)

    def test_rejects_bangla_only_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bangla.txt"
            path.write_text("বাংলা ভাষার মডেল প্রশিক্ষণ। " * 30, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Bilingual corpus check failed"):
                inspect_bilingual_corpus(path)

if __name__ == "__main__": unittest.main()
