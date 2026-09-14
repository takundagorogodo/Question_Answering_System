"""
Stage 18 - TF-IDF baseline: Question -> TF-IDF cosine -> stored chunk (NOT generative)

This is the baseline you compare AGAINST. Final system must be generative,
this baseline is allowed only as Stage 18 comparison.

Shows why generative RAG is better: baseline returns verbatim chunk, no rewriting.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from pathlib import Path

def load_chunks():
    p = Path("data/processed/chunks.jsonl")
    chunks = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def build_index(chunks):
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix

def retrieve_tfidf(question, vectorizer, matrix, chunks, top_k=3):
    q_vec = vectorizer.transform([question])
    scores = cosine_similarity(q_vec, matrix)[0]
    top_idx = scores.argsort()[::-1][:top_k]
    results = []
    for i in top_idx:
        results.append({"score": float(scores[i]), "chunk": chunks[i]})
    return results

def main():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks for TF-IDF baseline")
    vectorizer, matrix = build_index(chunks)

    tests = [
        "Explain tokenization in simple words.",
        "What is retrieval augmented generation?",
        "What is the best recipe for chocolate cake?",
    ]

    for q in tests:
        hits = retrieve_tfidf(q, vectorizer, matrix, chunks, top_k=2)
        print(f"\nQ: {q}")
        for h in hits:
            print(f"  [{h['score']:.3f}] {h['chunk']['source_file']} #{h['chunk']['chunk_index']}: {h['chunk']['text'][:100]}...")
        print(f"  Baseline answer (verbatim copy): {hits[0]['chunk']['text'][:200]}")

if __name__ == "__main__":
    main()
