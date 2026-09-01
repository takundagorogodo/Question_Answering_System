import faiss
import numpy as np

from src.preprocessing import load_documents
from src.chunking import chunk_documents
from src.embeddings import embed_texts


class VectorStore:
    """
    Wraps a FAISS index plus the metadata needed to map
    search results back to the original chunk text and source.
    """

    def __init__(self):
        self.index = None
        self.chunks = []  # keeps chunk dicts in the same order as vectors in the index

    def build(self, chunks):
        """
        Embed all chunks and build the FAISS index from scratch.
        """
        self.chunks = chunks
        texts = [c["text"] for c in chunks]

        embeddings = embed_texts(texts)
        embeddings_np = embeddings.cpu().numpy().astype("float32")

        # Normalize vectors so that inner product search behaves like cosine similarity
        faiss.normalize_L2(embeddings_np)

        dimension = embeddings_np.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # IP = inner product
        self.index.add(embeddings_np)

    def search(self, query, top_k=3):
        """
        Embed the query and return the top_k most similar chunks,
        each with its similarity score.
        """
        query_embedding = embed_texts([query])
        query_np = query_embedding.cpu().numpy().astype("float32")
        faiss.normalize_L2(query_np)

        scores, indices = self.index.search(query_np, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            chunk = self.chunks[idx]
            results.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "title": chunk["title"],
                "chunk_id": chunk["chunk_id"],
                "score": float(score)
            })

        return results


if __name__ == "__main__":
    docs = load_documents()
    chunks = chunk_documents(docs)

    store = VectorStore()
    store.build(chunks)

    test_question = "Why do transformer models use attention?"
    results = store.search(test_question, top_k=3)

    print(f"Question: {test_question}\n")
    print("Top matches:\n")
    for r in results:
        preview = r["text"][:120].replace("\n", " ")
        print(f"[score {r['score']:.4f}] {r['title']} ({r['chunk_id']})")
        print(f"  {preview}...\n")