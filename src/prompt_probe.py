"""
Stage 10b - prompt probe: 5 variants on SAME context, greedy decode.

Results you already got:
 A_current: "insufficient information" -> over-strict
 B_no_escape: correct answer, verbatim True
 C_soft_escape: "I don't know" -> over-refuses

We now add D and E which should keep B's success but reduce verbatim copy.
Run: python -m src.prompt_probe
"""
from src.retriever import retrieve
from src.generator import get_model

VARIANTS = {
    "A_current": (
        "Context:\n{context}\n\nQuestion: {question}\n\n"
        "Answer the question using only the provided context, in your own words.\n"
        "If the context does not contain enough information, answer exactly: insufficient information."
    ),
    "B_no_escape": (
        "Context:\n{context}\n\nQuestion: {question}\n\n"
        "Answer the question in one or two sentences using the information above."
    ),
    "C_soft_escape": (
        "Context:\n{context}\n\nQuestion: {question}\n\n"
        "Answer in one or two sentences using only the context above. "
        "If the context has no relevant information, reply: I don't know."
    ),
    "D_own_words": (
        "Context:\n{context}\n\nQuestion: {question}\n\n"
        "Using the context above, answer the question in your own words in one or two sentences."
    ),
    "E_no_copy": (
        "Context:\n{context}\n\nQuestion: {question}\n\n"
        "Answer the question using the context. Write in your own words, do not copy sentences directly."
    ),
}

def main() -> None:
    tok, model = get_model()
    q = "Explain tokenization in simple words."
    hits = retrieve(q, top_k=2)
    context = "\n\n".join(h["text"] for h in hits)
    print(f"Q: {q}\ncontext chars: {len(context)} | top score {hits[0]['score']:.3f}\n")
    for name, template in VARIANTS.items():
        prompt = template.format(context=context, question=q)
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
        text = tok.decode(
            model.generate(**inputs, max_new_tokens=128, do_sample=False)[0],
            skip_special_tokens=True,
        ).strip()
        copied = text in context
        print(f"--- {name} ---")
        print("A:", text)
        print(f"verbatim copy: {copied}\n")

if __name__ == "__main__":
    main()
