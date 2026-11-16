import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from core.bpe_tokenizer import BytePairTokenizer
    from core.transformer import AiHubTransformer, TransformerConfig
    from core.transformer_server import make_handler


@unittest.skipIf(torch is None, "PyTorch is required for server tests")
class TransformerServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = BytePairTokenizer().fit("hello Ai'Hub", vocab_size=256)
        config = TransformerConfig(vocab_size=cls.tokenizer.vocab_size,
                                   context_length=16, layers=1, heads=2,
                                   embedding_dim=16, feed_forward_dim=32)
        cls.model = AiHubTransformer(config).eval()
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0),
                                         make_handler(cls.tokenizer, cls.model))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = "http://127.0.0.1:%s" % cls.server.server_port

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_health(self):
        with urlopen(self.base + "/health", timeout=5) as response:
            payload = json.loads(response.read())
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["model"], "Ai'Hub Core 0.1 Transformer")

    def test_generate(self):
        request = Request(self.base + "/generate",
                          data=json.dumps({"prompt": "hello", "max_new_tokens": 2,
                                           "top_k": 4}).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=10) as response:
            payload = json.loads(response.read())
        self.assertTrue(payload["text"].startswith("hello"))

    def test_rejects_empty_prompt(self):
        request = Request(self.base + "/generate",
                          data=b'{"prompt":""}',
                          headers={"Content-Type": "application/json"})
        with self.assertRaises(HTTPError) as caught:
            urlopen(request, timeout=5)
        self.assertEqual(caught.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
