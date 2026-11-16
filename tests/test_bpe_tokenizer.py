import unittest

from core.bpe_tokenizer import BytePairTokenizer


class BytePairTokenizerTests(unittest.TestCase):
    def test_round_trip_ascii_and_unicode(self):
        source = "Ai'Hub builds its own model. বাংলা ভাষা."
        tokenizer = BytePairTokenizer(min_frequency=2).fit(source, vocab_size=320)
        self.assertEqual(tokenizer.decode(tokenizer.encode(source)), source)

    def test_serialization_round_trip(self):
        tokenizer = BytePairTokenizer().fit("abababab xyz xyz", vocab_size=280)
        restored = BytePairTokenizer.from_dict(tokenizer.to_dict())
        self.assertEqual(restored.decode(restored.encode("abab xyz")), "abab xyz")
        self.assertEqual(restored.vocab_size, tokenizer.vocab_size)

    def test_unseen_unicode_is_encodable(self):
        tokenizer = BytePairTokenizer().fit("hello world", vocab_size=270)
        text = "こんにちは"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_vocab_floor(self):
        with self.assertRaises(ValueError):
            BytePairTokenizer().fit("hello", vocab_size=100)


if __name__ == "__main__":
    unittest.main()
