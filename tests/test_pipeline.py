import json
import tempfile
import unittest
from pathlib import Path

from core.model import NGramModel
from core.tokenizer import WordTokenizer
from core.train import train


class TokenizerTests(unittest.TestCase):
    def test_round_trip(self):
        tokenizer = WordTokenizer().fit("hello, world!")
        self.assertEqual(tokenizer.decode(tokenizer.encode("hello, world!")), "hello, world!")

    def test_unknown_token(self):
        tokenizer = WordTokenizer().fit("known word")
        self.assertEqual(tokenizer.encode("missing"), [0])


class ModelTests(unittest.TestCase):
    def test_learns_observed_transition(self):
        model = NGramModel(smoothing=0).fit([[3, 4], [3, 4], [3, 4]])
        self.assertEqual(model.next_token(3), 4)

    def test_generation_length(self):
        model = NGramModel().fit([[1, 2, 3]])
        self.assertEqual(len(model.generate([1], max_new_tokens=5, seed=1)), 6)


class TrainingTests(unittest.TestCase):
    def test_train_writes_loadable_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "corpus.txt"
            target = root / "model.json"
            source.write_text("alpha beta\nalpha gamma\n", encoding="utf-8")
            stats = train(source, target)
            payload = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(payload["prototype"], "bigram_pipeline_test")
            self.assertEqual(stats["tokens"], 4)
            self.assertEqual(payload["model"]["type"], "bigram")


if __name__ == "__main__":
    unittest.main()
