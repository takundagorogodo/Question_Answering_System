"""
Stage 6 - Chunking: processed documents -> retrieval-sized chunks.

WHY we chunk at all:
  The retriever compares a QUESTION against pieces of text. A whole 3000-char
  document about nine ideas matches every question weakly (diluted similarity)
  and forces the generator to wade through irrelevant text. A chunk should
  hold ONE idea.

WHY ~500 chars:
  Too small -> the chunk loses the context needed to answer.
  Too large -> noise returns, and small generators have limited input size.
  ~500 chars is roughly one tight paragraph = one idea. A target, not a law;
  we will confirm it with retrieval measurements, not vibes.

WHY overlap:
  If a cut lands between an idea and its explanation, the next chunk starts
  cold. Repeating the last ~100 chars of the previous chunk stitches the seam.
"""
import json
import re

from src.config import PROCESSED_DIR, CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list:
    return [s for s in SENTENCE_SPLIT.split(text) if s.strip()]


def pack_pieces(pieces, max_chars: int) -> list:
    """Greedy packing: group strings until adding one more would pass
    max_chars. A single piece longer than max_chars is split on sentence
    boundaries first, so we never cut mid-sentence."""
    chunks, current, current_len = [], [], 0
    for piece in pieces:
        parts = [piece] if len(piece) <= max_chars else split_sentences(piece)
        for part in parts:
            add = len(part) + (2 if current else 0)   # +2 for the "\n\n" join
            if current and current_len + add > max_chars:
                chunks.append("\n\n".join(current))
                current, current_len, add = [], 0, len(part)
            current.append(part)
            current_len += add
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def with_overlap(chunks: list, overlap: int) -> list:
    """Prefix each chunk (except the first) with the tail of the previous
    chunk, cut at a word boundary so we never start mid-word."""
    if overlap <= 0 or not chunks:
        return chunks
    out = [chunks[0]]
    for prev, cur in zip(chunks, chunks[1:]):
        tail = prev[-overlap:]
        space = tail.find(" ")
        if space != -1:
            tail = tail[space + 1:]
        out.append(tail + " " + cur)
    return out


def main() -> None:
    docs_file = PROCESSED_DIR / "documents.jsonl"
    out_file = PROCESSED_DIR / "chunks.jsonl"

    records, total_before = [], 0
    with docs_file.open(encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            paragraphs = doc["text"].split("\n\n")
            total_before += len(paragraphs)

            chunks = pack_pieces(paragraphs, CHUNK_MAX_CHARS)
            chunks = with_overlap(chunks, CHUNK_OVERLAP_CHARS)

            for i, chunk_text in enumerate(chunks):
                records.append({
                    "chunk_id": f"{doc['doc_id']}#chunk{i:02d}",
                    "doc_id": doc["doc_id"],
                    "source_file": doc["source_file"],
                    "title": doc["title"],
                    "chunk_index": i,
                    "text": chunk_text,
                    "char_count": len(chunk_text),
                })

    with out_file.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    sizes = [r["char_count"] for r in records]
    print(f"Paragraphs in: {total_before}  ->  chunks out: {len(records)}")
    print(f"Chunk sizes: min {min(sizes)}, avg {sum(sizes)//len(sizes)}, max {max(sizes)}")
    print(f"Wrote {out_file}")

    # before/after example for the report: first document
    first = [r for r in records if r["doc_id"] == records[0]["doc_id"]]
    print(f"\nExample: '{first[0]['title']}' -> {len(first)} chunks")
    print(f"First chunk ({first[0]['char_count']} chars):")
    print(first[0]["text"][:300] + ("..." if first[0]["char_count"] > 300 else ""))


if __name__ == "__main__":
    main()