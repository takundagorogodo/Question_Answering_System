"""
Stage 5 - Preprocessing: corpus files -> cleaned, de-duplicated JSONL.

WHAT this does, and why nothing else:
  * whitespace / newline normalisation   -> stable text for embeddings
  * unicode normalisation (NFKC)         -> smart quotes etc. become plain ones
  * paragraph-level de-duplication       -> same paragraph twice wastes index space
  * metadata per document (source file, title) -> provenance for the Sources panel

WHAT this deliberately does NOT do (exam answer!):
  * no lowercasing of stored text  -> display and embeddings need original case;
    case-folding belongs to old keyword systems, not transformers
  * no stopword removal / stemming -> these DESTROY the semantic signal that
    modern embedding models rely on; legacy techniques from the keyword era
"""
import json
import re
import unicodedata
from pathlib import Path

from src.config import CORPUS_DIR, PROCESSED_DIR

MIN_PARAGRAPH_CHARS = 40   # shorter blocks are headings/fragments, not dedupe targets


def normalize_text(raw: str) -> str:
    """One clean, whitespace-stable version of the raw text."""
    text = unicodedata.normalize("NFKC", raw)      # unify smart quotes/dashes
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)            # collapse spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)         # at most one blank line
    return text.strip()


def split_paragraphs(text: str) -> list:
    """Split on blank lines; return non-empty stripped paragraphs."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def load_documents():
    """Yield (file_name, title, normalised_text, paragraphs) per corpus file."""
    for path in sorted(CORPUS_DIR.glob("*")):
        if path.name.startswith("_"):              # templates are not corpus
            continue
        if path.suffix.lower() not in (".txt", ".md"):
            continue
        raw = path.read_text(encoding="utf-8")
        text = normalize_text(raw)
        paragraphs = split_paragraphs(text)
        title = paragraphs[0] if paragraphs else path.stem
        yield path.name, title, text, paragraphs


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    seen = {}        # normalised paragraph -> first file that contained it
    duplicates = []  # (preview, this_file, first_file)
    records = []

    for file_name, title, text, paragraphs in load_documents():
        kept = []
        for para in paragraphs:
            key = re.sub(r"\s+", " ", para).lower()
            if len(key) >= MIN_PARAGRAPH_CHARS:
                if key in seen:
                    duplicates.append((para[:60], file_name, seen[key]))
                    continue
                seen[key] = file_name
            kept.append(para)

        records.append({
            "doc_id": Path(file_name).stem,
            "source_file": file_name,
            "title": title,
            "text": "\n\n".join(kept),
            "paragraphs": len(kept),
            "char_count": len(text),
        })

    out = PROCESSED_DIR / "documents.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Loaded {len(records)} documents from {CORPUS_DIR.name}/")
    print(f"Duplicates removed: {len(duplicates)}")
    for preview, this_file, first_file in duplicates:
        print(f"  - '{preview}...' in {this_file} (already in {first_file})")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
