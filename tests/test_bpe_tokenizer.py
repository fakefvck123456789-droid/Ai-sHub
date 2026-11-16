import unittest

from core.bpe_tokenizer import BytePairTokenizer


def reference_encode(text, merges):
    sequence = list(text.encode("utf-8"))
    for rank, pair in enumerate(merges):
        merged = []
        index = 0
        while index < len(sequence):
            if index + 1 < len(sequence) and (sequence[index], sequence[index + 1]) == pair:
                merged.append(256 + rank)
                index += 2
            else:
                merged.append(sequence[index])
                index += 1
        sequence = merged
    return sequence


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
        self.assertEqual(restored.to_dict(), tokenizer.to_dict())

    def test_unseen_unicode_is_encodable(self):
        tokenizer = BytePairTokenizer().fit("hello world", vocab_size=270)
        text = "こんにちは"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_ranked_encoder_matches_reference_merges(self):
        training = "aaaaaaa abababab xyzxyz " * 5
        tokenizer = BytePairTokenizer().fit(training, vocab_size=320)
        for sample in ("aaaaa", "ababababa", "xyz xyz", "বাংলা", "a" * 300):
            self.assertEqual(tokenizer.encode(sample), reference_encode(sample, tokenizer.merges))

    def test_training_sample_limit_is_serialized(self):
        tokenizer = BytePairTokenizer(max_training_bytes=64).fit("ab" * 1000, vocab_size=280)
        restored = BytePairTokenizer.from_dict(tokenizer.to_dict())
        self.assertEqual(restored.max_training_bytes, 64)

    def test_vocab_floor(self):
        with self.assertRaises(ValueError):
            BytePairTokenizer().fit("hello", vocab_size=100)

    def test_training_sample_limit_must_be_positive(self):
        with self.assertRaises(ValueError):
            BytePairTokenizer(max_training_bytes=0)


if __name__ == "__main__":
    unittest.main()
