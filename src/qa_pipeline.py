from src.preprocessing import load_documents
from src.chunking import chunk_documents
from src.retrieval import VectorStore
from src.generator import generate_answer

RELEVANCE_THRESHOLD = 0.35

# Build the vector store once, at import time.
# In the Streamlit app (Stage 15), this happens once when the app starts,
# not on every question - that would be far too slow.
_docs = load_documents()
_chunks = chunk_documents(_docs)
_store = VectorStore()
_store.build(_chunks)


def answer_question(question, top_k=3):
    """
    Full QA pipeline: retrieve -> check relevance -> generate (or decline).

    Returns a dict with:
      - answer: the generated answer, or a fallback message
      - grounded: True/False, whether the answer is backed by retrieved context
      - sources: list of retrieved chunks (empty if below threshold)
      - top_score: the best similarity score found
    """
    retrieved = _store.search(question, top_k=top_k)
    top_score = retrieved[0]["score"] if retrieved else 0.0

    if top_score < RELEVANCE_THRESHOLD:
        return {
            "answer": "I don't have enough information in my knowledge base to answer that confidently.",
            "grounded": False,
            "sources": [],
            "top_score": top_score
        }

    answer = generate_answer(question, retrieved)

    return {
        "answer": answer,
        "grounded": True,
        "sources": retrieved,
        "top_score": top_score
    }


if __name__ == "__main__":
    test_questions = [
        "What is the difference between extractive and generative QA?",  # in-domain
        "What is the capital of France?",                                # out-of-domain
        "How do I bake a chocolate cake?",                               # out-of-domain
    ]

    for q in test_questions:
        result = answer_question(q)
        print(f"Question: {q}")
        print(f"  Grounded: {result['grounded']} | Top score: {result['top_score']:.4f}")
        print(f"  Answer: {result['answer']}\n")