import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from core.bpe_tokenizer import BytePairTokenizer
    from core.router import LocalModelRouter, ModelNotFound, UnsupportedTask
    from core.transformer import AiHubTransformer, TransformerConfig


@unittest.skipIf(torch is None, "PyTorch is required for router tests")
class LocalModelRouterTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = BytePairTokenizer()
        self.model = AiHubTransformer(TransformerConfig(
            vocab_size=256, context_length=8, layers=1, heads=1,
            embedding_dim=16, feed_forward_dim=32,
        ))
        self.router = LocalModelRouter(self.tokenizer, self.model)

    def test_core_is_registered_without_external_fallback(self):
        self.assertFalse(self.router.external_fallback)
        self.assertEqual(self.router.resolve("text_generation").model_id, "core-0.1")

    def test_can_register_an_independent_local_family(self):
        code_model = AiHubTransformer(TransformerConfig(
            vocab_size=256, context_length=8, layers=1, heads=1,
            embedding_dim=16, feed_forward_dim=32,
        ))
        self.router.register("code-0.1", "code", ("code_generation",),
                             self.tokenizer, code_model)
        selected = self.router.resolve("code_generation", "code-0.1")
        self.assertEqual(selected.family, "code")
        self.assertEqual(selected.capabilities, ("code_generation",))
        self.assertIn("code-0.1", [item["id"] for item in self.router.list_models()])

    def test_duplicate_model_id_is_rejected(self):
        with self.assertRaises(ValueError):
            self.router.register("core-0.1", "code", ("code_generation",),
                                 self.tokenizer, self.model)

    def test_unavailable_family_does_not_fall_back(self):
        with self.assertRaises(UnsupportedTask):
            self.router.resolve("image_generation")
        with self.assertRaises(ModelNotFound):
            self.router.resolve("text_generation", "vision-0.1")


if __name__ == "__main__":
    unittest.main()
