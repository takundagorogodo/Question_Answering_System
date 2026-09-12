"""
Stage 7 - Embeddings: text -> vectors, plus the paraphrase demo.

WHAT an embedding is (30-second version):
  A sentence converted to a list of 384 numbers = coordinates in "meaning
  space". Sentences with similar meaning land CLOSE together; unrelated
  meanings land far apart. Closeness is measured with cosine similarity
  (1.0 = same direction, 0.0 = unrelated).

WHY keywords are not enough:
  "What is NLP?" and "Explain natural language processing." share almost no
  words, yet mean the same thing. Embeddings capture meaning; keyword
  counters cannot. This demo is your live-presentation moment.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL, MODELS_DIR

_MODEL = None


def get_model():
    """Load once, reuse everywhere. cache_folder keeps the weights inside the
    project (models/) so the demo machine needs no internet after first run."""
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(EMBEDDING_MODEL, cache_folder=str(MODELS_DIR))
    return _MODEL


def embed(texts):
    """list[str] -> (N, 384) numpy array of NORMALISED vectors.
    Normalised = each vector has length 1, so a dot product IS cosine
    similarity. We rely on this in the retriever (Stage 9)."""
    return get_model().encode(list(texts), normalize_embeddings=True,
                              show_progress_bar=False)


def cosine(a, b) -> float:
    return float(np.dot(a, b))


def main() -> None:
    s1 = "What is NLP?"
    s2 = "Explain natural language processing."
    s3 = "How do I make paneer at home?"

    vecs = embed([s1, s2, s3])
    print(f"vector dimensions: {vecs.shape[1]}")
    print(f"cosine('{s1}', '{s2}') = {cosine(vecs[0], vecs[1]):.3f}   <- paraphrases: expect HIGH")
    print(f"cosine('{s1}', '{s3}') = {cosine(vecs[0], vecs[2]):.3f}   <- unrelated: expect LOW")
    print("Read it as: same meaning -> close to 1; unrelated -> close to 0.")


if __name__ == "__main__":
    main()
