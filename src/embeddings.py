from sentence_transformers import SentenceTransformer, util

# Load the embedding model once, at import time, so we don't reload it
# every time we need to embed something (that would be slow).
_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(texts):
    """
    Convert a list of strings into a list of embedding vectors.
    """
    return _model.encode(texts, convert_to_tensor=True)


def cosine_similarity(vec1, vec2):
    """
    Return the cosine similarity between two embedding vectors (0-1 range, roughly).
    """
    return util.cos_sim(vec1, vec2).item()


if __name__ == "__main__":
    # Prove that paraphrased questions land close together in vector space.
    pairs = [
        ("What is NLP?", "Explain natural language processing."),
        ("What is NLP?", "What's your favorite pizza topping?"),
        ("How does chunking work?", "Why do we split documents into smaller pieces?"),
        ("How does chunking work?", "What is the capital of France?"),
    ]

    for text_a, text_b in pairs:
        emb_a = embed_texts([text_a])
        emb_b = embed_texts([text_b])
        score = cosine_similarity(emb_a[0], emb_b[0])
        print(f"'{text_a}'  <->  '{text_b}'\n  similarity: {score:.4f}\n")