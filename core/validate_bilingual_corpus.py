"""Check that a UTF-8 corpus contains meaningful Bangla and English text."""
from __future__ import annotations
import argparse, json, unicodedata
from pathlib import Path

def inspect_bilingual_corpus(path: Path, minimum_share: float = 0.10) -> dict:
    if not 0 < minimum_share < 0.5:
        raise ValueError("minimum_share must be greater than 0 and less than 0.5")
    text = path.read_text(encoding="utf-8")
    bangla = latin = letters = 0
    for char in text:
        if not unicodedata.category(char).startswith("L"):
            continue
        letters += 1
        code = ord(char)
        if 0x0980 <= code <= 0x09FF: bangla += 1
        elif 0x0041 <= code <= 0x005A or 0x0061 <= code <= 0x007A: latin += 1
    if letters == 0: raise ValueError("Corpus contains no letters")
    result = {"file": str(path), "letter_count": letters, "bangla_letters": bangla,
              "english_latin_letters": latin, "bangla_share": bangla / letters,
              "english_share": latin / letters, "minimum_share": minimum_share,
              "balanced_enough": bangla / letters >= minimum_share and latin / letters >= minimum_share}
    if not result["balanced_enough"]:
        raise ValueError("Bilingual corpus check failed: both Bangla and English must each make up "
                         f"at least {minimum_share:.0%} of letters (Bangla {bangla / letters:.1%}, "
                         f"English {latin / letters:.1%}).")
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--minimum-share", type=float, default=0.10)
    args = parser.parse_args()
    print(json.dumps(inspect_bilingual_corpus(args.input, args.minimum_share), ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
