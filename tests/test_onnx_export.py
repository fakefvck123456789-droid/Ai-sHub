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
    from core.export_onnx import export_checkpoint
    from core.transformer import AiHubTransformer, TransformerConfig


@unittest.skipIf(torch is None, "PyTorch, ONNX and ONNX Runtime are required")
class OnnxExportTests(unittest.TestCase):
    def test_dynamic_sequence_export_matches_pytorch(self):
        config = TransformerConfig(vocab_size=32, context_length=16, layers=1,
                                   heads=2, embedding_dim=16, feed_forward_dim=32)
        model = AiHubTransformer(config)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "tiny.pt"
            output = Path(directory) / "tiny.onnx"
            torch.save({
                "format_version": 1,
                "project": "Ai'Hub Core 0.1",
                "config": model.config_dict(),
                "model_state_dict": model.state_dict(),
            }, checkpoint)
            result = export_checkpoint(checkpoint, output)
            self.assertTrue(result["verified"])
            self.assertLessEqual(result["max_absolute_error"], 1e-4)
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
