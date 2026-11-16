# Ai'Hub architecture — foundation stage

## Scope

This repository begins with a small, inspectable baseline to prove the end-to-end data and inference path. It is not a neural language model and must not be represented as one.

## Pipeline today

1. Read UTF-8 text supplied by the developer.
2. Build a deterministic word/punctuation vocabulary.
3. Count unigram and adjacent-token (bigram) frequencies.
4. Serialize tokenizer and counts as JSON.
5. Load the artifact in a local HTTP server and generate by sampling observed transitions.

The model has no learned dense parameters, embeddings, attention, gradient descent, or semantic reasoning.

## Core 0.1 target

The planned first neural model is a decoder-only Transformer with context length 512, 4 blocks, 4 attention heads, 256 hidden dimensions, 1024-dimensional feed-forward layers, RoPE, RMSNorm, causal self-attention, and next-token cross-entropy training.

Implementation sequence:
1. Establish corpus provenance and deterministic data preparation.
2. Implement and test a custom BPE tokenizer.
3. Implement the Transformer and causal mask with numerical tests.
4. Add checkpointing, evaluation, and reproducible training configuration.
5. Benchmark CPU memory and throughput before attempting Android deployment.

## Runtime security

The HTTP server binds to localhost by default. Do not expose it to a public network without authentication, request limits, and a security review.
