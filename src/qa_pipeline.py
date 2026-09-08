from src.preprocessing import load_documents
from src.chunking import chunk_documents
from src.retrieval import VectorStore
from src.generator import generate_answer

RELEVANCE_THRESHOLD = 0.35

# Build the vector store once, at import time.
# In the Streamlit app, this happens once when the app starts,
# not on every question - that would be far too slow.
_docs = load_documents()
_chunks = chunk_documents(_docs)
_store = VectorStore()
_store.build(_chunks)


def answer_question(question, top_k=3):
    """
    Full QA pipeline: retrieve -> check relevance -> filter weak chunks -> generate (or decline).

    Returns a dict with:
      - answer: the generated answer, or a fallback message
      - grounded: True/False, whether the answer is backed by retrieved context
      - sources: list of retrieved chunks actually used (empty if below threshold)
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

    # Only keep chunks that individually clear the threshold, so weak/irrelevant
    # chunks don't get fed into the generator as noise alongside the good ones.
    relevant_chunks = [r for r in retrieved if r["score"] >= RELEVANCE_THRESHOLD]

    answer = generate_answer(question, relevant_chunks)

    return {
        "answer": answer,
        "grounded": True,
        "sources": relevant_chunks,
        "top_score": top_score
    }


if __name__ == "__main__":
    test_questions = [
        "What is the difference between extractive and generative QA?",
        "What is RAG?",
        "What is the capital of France?",
    ]

    for q in test_questions:
        result = answer_question(q)
        print(f"Question: {q}")
        print(f"  Grounded: {result['grounded']} | Top score: {result['top_score']:.4f}")
        print(f"  Sources used: {len(result['sources'])}")
        print(f"  Answer: {result['answer']}\n")