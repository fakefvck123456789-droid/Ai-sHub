import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from core.transformer_generate import load_checkpoint, generate
    from core.transformer_train import train
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required for training tests")
class TransformerTrainingTests(unittest.TestCase):
    def test_train_save_load_and_generate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "tiny.txt"
            checkpoint = root / "checkpoint.pt"
            corpus.write_text(("Ai'Hub builds independent models. " * 20), encoding="utf-8")
            stats = train(corpus, checkpoint, steps=1, batch_size=2,
                          context_length=8, vocab_size=256)
            self.assertTrue(checkpoint.exists())
            self.assertEqual(stats["steps"], 1)
            model, tokenizer = load_checkpoint(checkpoint)
            self.assertEqual(model.config.vocab_size, tokenizer.vocab_size)
            generated = generate(checkpoint, "Ai", max_new_tokens=2, top_k=5)
            self.assertTrue(generated.startswith("Ai"))


if __name__ == "__main__":
    unittest.main()
