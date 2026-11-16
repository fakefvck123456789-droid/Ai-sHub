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


if __name__ == "__main__":
    unittest.main()
