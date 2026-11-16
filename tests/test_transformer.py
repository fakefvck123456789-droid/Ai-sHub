import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from core.transformer import AiHubTransformer, RMSNorm, TransformerConfig, apply_rope


@unittest.skipIf(torch is None, "PyTorch is required for Transformer tests")
class TransformerTests(unittest.TestCase):
    def make_model(self):
        config = TransformerConfig(vocab_size=32, context_length=16, layers=2,
                                   heads=4, embedding_dim=32, feed_forward_dim=64)
        return AiHubTransformer(config).eval()

    def test_forward_shape_and_loss(self):
        model = self.make_model()
        x = torch.randint(0, 32, (2, 8))
        logits, loss = model(x, x)
        self.assertEqual(tuple(logits.shape), (2, 8, 32))
        self.assertTrue(torch.isfinite(loss).item())

    def test_rejects_long_context(self):
        model = self.make_model()
        with self.assertRaises(ValueError):
            model(torch.ones((1, 17), dtype=torch.long))

    def test_generation_appends_requested_tokens(self):
        model = self.make_model()
        prompt = torch.tensor([[1, 2, 3]])
        generated = model.generate(prompt, max_new_tokens=4, top_k=5)
        self.assertEqual(tuple(generated.shape), (1, 7))

    def test_rope_preserves_shape_and_norm(self):
        x = torch.randn(2, 4, 7, 8)
        rotated = apply_rope(x, torch.arange(7))
        self.assertEqual(tuple(rotated.shape), tuple(x.shape))
        self.assertTrue(torch.allclose(x.norm(dim=-1), rotated.norm(dim=-1), atol=1e-5))

    def test_rmsnorm_shape(self):
        layer = RMSNorm(12)
        self.assertEqual(tuple(layer(torch.randn(2, 3, 12)).shape), (2, 3, 12))

    def test_future_tokens_do_not_change_earlier_logits(self):
        model = self.make_model()
        first = torch.tensor([[1, 2, 3, 4]])
        changed = torch.tensor([[1, 2, 9, 8]])
        with torch.no_grad():
            a, _ = model(first)
            b, _ = model(changed)
        self.assertTrue(torch.allclose(a[:, :2], b[:, :2], atol=1e-6))

    def test_input_embedding_and_output_head_are_tied(self):
        model = self.make_model()
        self.assertIs(model.token_embedding.weight, model.lm_head.weight)


if __name__ == "__main__":
    unittest.main()
