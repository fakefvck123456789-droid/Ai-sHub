import tempfile
import unittest
from pathlib import Path

try:
    import torch
    import onnx
    import onnxruntime
except ImportError:
    torch = None

if torch is not None:
    from core.bpe_tokenizer import BytePairTokenizer
    from core.export_onnx import export_checkpoint
    from core.transformer import AiHubTransformer, TransformerConfig
    from core.transformer_train import train


@unittest.skipIf(torch is None, "PyTorch, ONNX and ONNX Runtime are required")
class OnnxExportTests(unittest.TestCase):
    def test_dynamic_sequence_export_matches_pytorch(self):
        tokenizer = BytePairTokenizer().fit(
            "Ai'Hub local model export test. " * 8, vocab_size=256
        )
        config = TransformerConfig(vocab_size=tokenizer.vocab_size, context_length=16,
                                   layers=1, heads=2, embedding_dim=16,
                                   feed_forward_dim=32)
        model = AiHubTransformer(config)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "tiny.pt"
            output = Path(directory) / "tiny.onnx"
            torch.save({
                "format_version": 1,
                "project": "Ai'Hub Core 0.1",
                "config": model.config_dict(),
                "tokenizer": tokenizer.to_dict(),
                "model_state_dict": model.state_dict(),
            }, checkpoint)
            result = export_checkpoint(checkpoint, output)
            self.assertTrue(result["verified"])
            self.assertLessEqual(result["max_absolute_error"], 1e-4)
            self.assertTrue(output.exists())
            self.assertTrue(Path(result["tokenizer"]).exists())
            self.assertTrue(Path(result["metadata"]).exists())

    def test_real_training_checkpoint_exports_for_android(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.txt"
            checkpoint = root / "trained.pt"
            onnx_path = root / "core-0.1.onnx"
            corpus.write_text(
                ("Ai'Hub builds local models. This is a training pipeline test.\n" * 120),
                encoding="utf-8",
            )
            stats = train(
                corpus, checkpoint, steps=2, batch_size=2,
                context_length=16, vocab_size=256, eval_interval=1,
                eval_batches=1, validation_fraction=0.2, seed=7,
            )
            self.assertTrue(checkpoint.exists())
            self.assertEqual(stats["steps"], 2)
            result = export_checkpoint(checkpoint, onnx_path)
            self.assertTrue(result["verified"])
            self.assertTrue(onnx_path.exists())
            metadata = __import__("json").loads(Path(result["metadata"]).read_text())
            self.assertEqual(metadata["config"]["vocab_size"], result["vocabulary"])
            self.assertTrue(Path(result["tokenizer"]).exists())


if __name__ == "__main__":
    unittest.main()
