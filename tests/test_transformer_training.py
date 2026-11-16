import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from core.transformer_eval import evaluate_checkpoint
    from core.transformer_generate import load_checkpoint, generate
    from core.transformer_train import train


@unittest.skipIf(torch is None, "PyTorch is required for training tests")
class TransformerTrainingTests(unittest.TestCase):
    def test_train_save_load_and_generate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "tiny.txt"
            checkpoint = root / "checkpoint.pt"
            corpus.write_text(("Ai'Hub builds independent models. " * 20), encoding="utf-8")
            stats = train(corpus, checkpoint, steps=1, batch_size=2,
                          context_length=8, vocab_size=256, eval_batches=2)
            self.assertTrue(checkpoint.exists())
            self.assertEqual(stats["steps"], 1)
            self.assertGreater(stats["training_tokens"], 0)
            self.assertGreater(stats["validation_tokens"], 0)
            self.assertTrue(torch.isfinite(torch.tensor(stats["validation_loss"])).item())
            model, tokenizer = load_checkpoint(checkpoint)
            self.assertEqual(model.config.vocab_size, tokenizer.vocab_size)
            self.assertEqual(model.config.layers, 4)
            self.assertEqual(model.config.heads, 4)
            self.assertEqual(model.config.embedding_dim, 256)
            self.assertEqual(model.config.feed_forward_dim, 1024)
            generated = generate(checkpoint, "Ai", max_new_tokens=2, top_k=5)
            self.assertTrue(generated.startswith("Ai"))


    def test_resume_continues_from_saved_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "tiny.txt"
            checkpoint = root / "checkpoint.pt"
            corpus.write_text(("Ai'Hub builds independent models. " * 20), encoding="utf-8")
            train(corpus, checkpoint, steps=1, batch_size=2,
                  context_length=8, vocab_size=256, eval_batches=1)
            resumed = train(corpus, checkpoint, steps=2, batch_size=2,
                            context_length=8, vocab_size=256, eval_batches=1,
                            resume_from=checkpoint)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(resumed["steps"], 2)
            self.assertEqual(payload["training"]["completed_steps"], 2)
            self.assertIn("optimizer_state_dict", payload)


    def test_standalone_evaluation_reports_perplexity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "train.txt"
            evaluation = root / "evaluation.txt"
            checkpoint = root / "checkpoint.pt"
            corpus.write_text(("Ai'Hub builds independent models. " * 20), encoding="utf-8")
            evaluation.write_text("A separate evaluation corpus.", encoding="utf-8")
            train(corpus, checkpoint, steps=1, batch_size=2,
                  context_length=8, vocab_size=256, eval_batches=1)
            metrics = evaluate_checkpoint(checkpoint, evaluation)
            self.assertEqual(metrics["target_tokens"], metrics["evaluation_tokens"] - 1)
            self.assertGreater(metrics["perplexity"], 0)
            self.assertTrue(torch.isfinite(torch.tensor(metrics["average_loss"])).item())

    def test_gradient_accumulation_and_cosine_schedule_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "tiny.txt"
            checkpoint = root / "checkpoint.pt"
            corpus.write_text(("Ai'Hub builds independent models. " * 30), encoding="utf-8")
            stats = train(
                corpus, checkpoint, steps=3, batch_size=1,
                context_length=8, vocab_size=256, eval_batches=1,
                gradient_accumulation_steps=2, warmup_steps=1,
                min_lr_ratio=0.2,
            )
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            training = payload["training"]
            self.assertEqual(stats["effective_batch_size"], 2)
            self.assertEqual(training["gradient_accumulation_steps"], 2)
            self.assertEqual(training["effective_batch_size"], 2)
            self.assertEqual(training["warmup_steps"], 1)
            self.assertAlmostEqual(training["current_learning_rate"], 3e-4 * 0.2, places=8)

    def test_invalid_scheduler_settings_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "tiny.txt"
            corpus.write_text(("Ai'Hub builds independent models. " * 20), encoding="utf-8")
            with self.assertRaises(ValueError):
                train(corpus, root / "unused.pt", steps=2, context_length=8,
                      vocab_size=256, warmup_steps=2)
            with self.assertRaises(ValueError):
                train(corpus, root / "unused.pt", steps=2, context_length=8,
                      vocab_size=256, min_lr_ratio=1.5)

    def test_rejects_too_short_validation_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "short.txt"
            corpus.write_text("too short", encoding="utf-8")
            with self.assertRaises(ValueError):
                train(corpus, root / "unused.pt", steps=1, context_length=8,
                      vocab_size=256)


if __name__ == "__main__":
    unittest.main()
