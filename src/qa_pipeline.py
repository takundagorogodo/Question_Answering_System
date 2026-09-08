RELEVANCE_THRESHOLD = 0.35


def answer_question(question, top_k=3):
    retrieved = _store.search(question, top_k=top_k)
    top_score = retrieved[0]["score"] if retrieved else 0.0

    if top_score < RELEVANCE_THRESHOLD:
        return {
            "answer": "I don't have enough information in my knowledge base to answer that confidently.",
            "grounded": False,
            "sources": [],
            "top_score": top_score
        }

    # NEW: only keep chunks that individually clear the threshold,
    # so weak/irrelevant chunks don't get fed into the generator as noise.
    relevant_chunks = [r for r in retrieved if r["score"] >= RELEVANCE_THRESHOLD]

    answer = generate_answer(question, relevant_chunks)

    return {
        "answer": answer,
        "grounded": True,
        "sources": relevant_chunks,
        "top_score": top_score
    }