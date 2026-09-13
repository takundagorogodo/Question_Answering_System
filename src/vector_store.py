"""
Stage 8 - Vector store: chunks + embeddings -> FAISS index.

WHAT a vector store is:
  A saved copy of every chunk's 384-dim vector so we can ask "which vectors
  are closest to this query vector?" very fast. FAISS is the standard local
  option: no server, one pip package, works offline.

WHY IndexFlatIP:
  Our vectors are NORMALISED (Stage 7), so inner product == cosine similarity.
  "Flat" = exact search, no approximation. For a few hundred chunks exact
  search is instant; approximate indexes (IVF/HNSW) only matter at millions.
  "We chose exact search because the corpus is small" is a defensible
  engineering decision - say it in the viva.

NOTE: the index files are git-ignored on purpose - they are derived artefacts,
rebuilt from data/processed by this script.
"""
import json
import pickle

import faiss

from src.config import INDEX_DIR, PROCESSED_DIR
from src.embeddings import embed

INDEX_FILE = INDEX_DIR / "faiss.index"
META_FILE = INDEX_DIR / "chunks_meta.pkl"


def build_index():
    """Embed every chunk, store vectors in FAISS, store chunk metadata in the
    same order (row i of the index == chunks[i])."""
    with open(PROCESSED_DIR / "chunks.jsonl", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    vectors = embed([c["text"] for c in chunks]).astype("float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))
    with open(META_FILE, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Indexed {index.ntotal} chunks of dim {vectors.shape[1]}")
    print(f"Saved {INDEX_FILE}")
    print(f"Saved {META_FILE}")
    return index, chunks


def load_index():
    """The retriever (Stage 9) uses this: index + metadata, order aligned."""
    index = faiss.read_index(str(INDEX_FILE))
    with open(META_FILE, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def main() -> None:
    build_index()


if __name__ == "__main__":
    main()