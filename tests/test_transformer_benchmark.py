import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from core.transformer_benchmark import benchmark
    from core.transformer_train import train


@unittest.skipIf(torch is None, "PyTorch is required for benchmark tests")
class TransformerBenchmarkTests(unittest.TestCase):
    def test_benchmark_reports_positive_throughput(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "tiny.txt"
            checkpoint = root / "checkpoint.pt"
            corpus.write_text(("Ai'Hub builds independent models. " * 30), encoding="utf-8")
            train(corpus, checkpoint, steps=1, batch_size=1,
                  context_length=8, vocab_size=256, eval_batches=1)
            result = benchmark(checkpoint, "Ai", new_tokens=2, warmup=0, repeats=1)
            self.assertGreater(result["latency_seconds_mean"], 0)
            self.assertGreater(result["tokens_per_second_mean"], 0)
            self.assertEqual(result["generated_tokens_per_run"], 2)
            self.assertGreater(result["parameters"], 0)

    def test_benchmark_rejects_invalid_repeat_count(self):
        with self.assertRaises(ValueError):
            benchmark(Path("missing.pt"), "Ai", repeats=0)


if __name__ == "__main__":
    unittest.main()
