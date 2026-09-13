"""
Stage 14 (early) + Stage 13 hallucination gate - the CORE pipeline.
Question -> retrieve -> threshold check -> generate -> sources + confidence.

This is what Stage 15 Streamlit UI will call. No UI code here.
"""
import time
from src.config import RETRIEVAL_TOP_K, SIMILARITY_THRESHOLD
from src.retriever import retrieve
from src.generator import generate

def answer_question(question: str, top_k: int = RETRIEVAL_TOP_K, threshold: float = SIMILARITY_THRESHOLD) -> dict:
    """
    Returns dict:
      question, answer, sources (with scores), top_score, confidence,
      grounded (bool), latency_sec
    """
    start = time.time()
    hits = retrieve(question, top_k=top_k)
    if not hits:
        return {
            "question": question,
            "answer": "insufficient information",
            "sources": [],
            "top_score": 0.0,
            "confidence": "low",
            "grounded": False,
            "latency_sec": time.time() - start,
        }

    top_score = hits[0]["score"]
    context = "\n\n".join(h["text"] for h in hits)

    # Stage 13 gate: if retrieval is weak, DO NOT call generator
    if top_score < threshold:
        return {
            "question": question,
            "answer": "insufficient information - no relevant context found",
            "sources": hits,
            "top_score": top_score,
            "confidence": "low",
            "grounded": False,
            "latency_sec": time.time() - start,
        }

    answer = generate(question, context)

    # confidence from top_score
    if top_score >= 0.6:
        conf = "high"
    elif top_score >= 0.4:
        conf = "medium"
    else:
        conf = "low"

    # grounded check: is answer supported? simple heuristic for now
    # True if not refusing and top_score >= threshold
    grounded = top_score >= threshold and "insufficient" not in answer.lower() and "i don't know" not in answer.lower()

    return {
        "question": question,
        "answer": answer,
        "sources": hits,
        "top_score": top_score,
        "confidence": conf,
        "grounded": grounded,
        "latency_sec": time.time() - start,
    }

def main() -> None:
    tests = [
        "Explain tokenization in simple words.",
        "What is retrieval augmented generation?",
        "What is hallucination in language models?",
        "What is the best recipe for chocolate cake?",  # OOD - should trigger low confidence
    ]
    for q in tests:
        r = answer_question(q)
        print(f"\nQ: {r['question']}")
        print(f"  top_score={r['top_score']:.3f} conf={r['confidence']} grounded={r['grounded']} time={r['latency_sec']:.2f}s")
        print(f"  A: {r['answer']}")
        print(f"  sources: {[h['source_file']+'#'+str(h['chunk_index']) for h in r['sources']]}")

if __name__ == "__main__":
    main()
