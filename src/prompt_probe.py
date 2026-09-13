"""
Stage 10b - prompt probe: the model refused to answer a question it clearly had
context for. Stage 10 needs ONE real generated answer before it counts as done,
so we test three prompt phrasings on the SAME retrieved context and keep the
winner. This is a preview of Stage 12 (prompt design), done on evidence instead
of intuition. Greedy decoding (do_sample=False) makes the comparison fair.
"""
from src.retriever import retrieve
from src.generator import get_model, PROMPT

VARIANTS = {
    "A_current": PROMPT,
    "B_no_escape": (
        "Context:\n{context}\n\nQuestion: {question}\n\n"
        "Answer the question in one or two sentences using the information above."
    ),
    "C_soft_escape": (
        "Context:\n{context}\n\nQuestion: {question}\n\n"
        "Answer in one or two sentences using only the context above. "
        "If the context has no relevant information, reply: I don't know."
    ),
}


def main() -> None:
    tok, model = get_model()

    q = "Explain tokenization in simple words."
    context = "\n\n".join(h["text"] for h in retrieve(q, top_k=2))
    print(f"Q: {q}\ncontext chars: {len(context)}\n")

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
        print(f"verbatim copy of context: {copied}\n")


if __name__ == "__main__":
    main()
