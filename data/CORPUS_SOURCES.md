# Corpus provenance

Every training dataset must be reviewed before use.

Record one entry per dataset:
- Dataset name and version
- Original source URL
- Copyright holder / creator, if known
- License name and canonical license URL
- Date obtained
- Whether commercial use, modification, and model training are permitted
- Any attribution, redistribution, or share-alike obligations
- Processing performed and files derived from it

Do not add scraped, private, confidential, or otherwise unlicensed text. The bundled `sample.txt` is synthetic text written for this repository's smoke test; it is not a real-world training corpus.

## Machine-readable source manifest

The preparation script requires a JSON object keyed by each input `.txt` file's path relative to the input directory. Every entry must contain:
- `source_url`: original dataset or source location
- `license`: exact license name
- `training_allowed`: boolean that must be explicitly set to `true`

Example structure (replace all placeholders with verified facts; this is not a dataset recommendation):

```json
{
  "dataset-a/part-001.txt": {
    "source_url": "https://verified-source.example/dataset",
    "license": "EXACT-LICENSE-NAME",
    "training_allowed": true
  }
}
```

Prepare a corpus with:

```bash
python -m core.prepare_corpus --input-dir data/raw --sources data/sources.json --output data/prepared/corpus.txt
```

The script normalizes Unicode to NFC, joins wrapped lines within paragraphs, removes duplicate paragraphs, and writes a SHA-256/provenance manifest beside the output. It does not determine whether a license legally permits model training; the person adding the source must verify that independently. Do not include private, confidential, or personal data without a valid legal basis and appropriate consent.
