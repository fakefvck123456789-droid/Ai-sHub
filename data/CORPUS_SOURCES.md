# Ai'Hub Core 0.1 — bilingual training data policy

Core 0.1 targets **Bangla (bn) and English (en)** together. The byte-level BPE tokenizer can represent both scripts, but a tokenizer alone does not make a model bilingual: the training corpus must contain substantial, high-quality examples in both languages.

## Candidate sources requiring review

These are candidate sources, not automatically approved training inputs. Review the exact dataset card, source-level rights, attribution, version, and local legal obligations before use.

| Candidate | Language | Published license / conditions | Notes |
|---|---|---|---|
| [B-CORE Bengali corpus](https://huggingface.co/datasets/nahid-hub/B-CORE-bengali-corpus) | Bangla | Dataset card states CC BY 4.0 and warns that individual source components may have separate attribution requirements | Reported 20.8 GB compressed. Too large to blindly download to a phone. Inspect source data and retain component attribution. |
| [Bangla NLP Corpus](https://huggingface.co/datasets/irfanhossainsust/bangla-nlp-corpus) | Bangla | Dataset card states CC BY-SA 4.0 | Card describes a 256,025-document validation release sourced from Bangla Wikipedia. Preserve share-alike and attribution obligations; verify source records. |
| [English Wikipedia dumps](https://dumps.wikimedia.org/enwiki/) | English | Wikimedia project text is generally offered under CC BY-SA 4.0 and GFDL, subject to terms and third-party exceptions | Use official dumps and preserve attribution/reuse notices. |
| [Bangla Wikipedia dumps](https://dumps.wikimedia.org/bnwiki/) | Bangla | Wikimedia project text is generally offered under CC BY-SA 4.0 and GFDL, subject to terms and third-party exceptions | Use official dumps and preserve attribution/reuse notices. |

Wikimedia terms say contributor text is licensed under CC BY-SA 4.0 and GFDL. This is not blanket clearance for every embedded third-party item or every downstream model distribution. Project Gutenberg is not treated as an automatically approved English corpus: its terms vary by work and are US-centered, so assess each title and Bangladesh distribution implications individually.

## Required workflow

1. Download only from official dataset/dump distribution.
2. Preserve dataset cards, license text, source URLs, version, retrieval date, and source-level attribution.
3. Do not assume a dataset-level license overrides licenses of component sources.
4. Exclude personal, private, confidential, paywalled, or otherwise unauthorized content.
5. Convert approved text to UTF-8 .txt files. Keep Bangla and English sources in separate folders.
6. Create data/sources.json with one entry per text file. Each entry needs source_url, exact license, and training_allowed: true. This records a human review; it is not legal advice.
7. Prepare and deduplicate, then validate both language shares:

    python -m core.prepare_corpus --input-dir data/raw --sources data/sources.json --output data/prepared/core-0.1.txt
    python -m core.validate_bilingual_corpus --input data/prepared/core-0.1.txt --minimum-share 0.10

The validator requires Bangla Unicode letters and basic Latin A–Z letters to each represent at least 10% of letters. This is only a mix guard, not a quality, dialect, or license check. Release evaluation must use separate Bangla and English held-out sets.

## Model policy

- Train one shared Core model and tokenizer on both languages.
- Do not translate Bangla into English as a substitute for native Bangla training.
- Keep separate held-out evaluation sets for Bangla and English.
- Report per-language loss/perplexity, generation examples, script integrity, and human review.
- Never describe a smoke-test checkpoint as a trained release model.
