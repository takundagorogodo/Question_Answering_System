"""
Stage 9 - Retriever: question -> top-K chunks + scores.

WHAT happens here:
  1. embed the question with the SAME model used for the chunks (mandatory -
     two different models would live in two different meaning-spaces)
  2. ask FAISS for the K nearest vectors (inner product = cosine, because
     all vectors are normalised)
  3. return the chunk texts WITH their scores

WHY the scores are shown:
  Transparency is the differentiator. "similarity 0.61, file 03, chunk 2" is
  auditable; a bare answer is not. These same numbers become the confidence
  signal in Stage 13 and the Sources panel in Stage 15.
"""
from src.config import RETRIEVAL_TOP_K
from src.embeddings import embed
from src.vector_store import load_index


def retrieve(question: str, top_k: int = RETRIEVAL_TOP_K) -> list:
    """Best-first list of {chunk fields + 'score'} dicts."""
    index, chunks = load_index()
    q = embed([question]).astype("float32")
    scores, ids = index.search(q, top_k)

    results = []
    for score, i in zip(scores[0], ids[0]):
        if i == -1:                      # FAISS pads with -1 if fewer hits than k
            continue
        rec = dict(chunks[i])
        rec["score"] = float(score)
        results.append(rec)
    return results


def main() -> None:
    demos = [
        "What is natural language processing?",
        "how does a machine cut text into little pieces?",   # paraphrase: tokenization
        "What is the best recipe for chocolate cake?",       # out of domain
    ]
    for q in demos:
        print(f"\nQ: {q}")
        for r in retrieve(q):
            snippet = r["text"][:80].replace("\n", " ")
            print(f"  [{r['score']:.3f}] {r['source_file']} #{r['chunk_index']}: {snippet}...")


if __name__ == "__main__":
    main()