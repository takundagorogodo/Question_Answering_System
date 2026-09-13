"""
Stage 10 - Generative QA: (question + retrieved context) -> WRITTEN answer.

WHY this is "generative" (and how you prove it):
  The model does NOT copy a span of the context. It reads context + question
  and decodes a new sentence token by token. The demo at the bottom prints a
  verbatim-copy check: it must say False, because the answer text should not
  exist anywhere in the context.

MODEL: FLAN-T5 - an encoder-decoder transformer, instruction-tuned.
  The encoder reads the whole prompt; the decoder writes the answer one token
  at a time, each token conditioned on everything before it. Small enough for
  CPU; explainable in a viva. ADR-001: benchmarked against a modern
  decoder-only model in Stage 11 - this module's interface stays the same
  whichever wins.
"""
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.config import GENERATOR_MODEL, MODELS_DIR

_TOKENIZER = None
_MODEL = None

PROMPT = """Context:
{context}

Question: {question}

Answer the question using only the provided context, in your own words.
If the context does not contain enough information, answer exactly: insufficient information."""


def get_model():
    """Load once, reuse. Weights cached inside the project (offline after 1st run)."""
    global _TOKENIZER, _MODEL
    if _MODEL is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(GENERATOR_MODEL,
                                                   cache_dir=str(MODELS_DIR))
        _MODEL = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_MODEL,
                                                       cache_dir=str(MODELS_DIR))
    return _TOKENIZER, _MODEL


def generate(question: str, context: str, max_new_tokens: int = 128) -> str:
    tok, model = get_model()
    prompt = PROMPT.format(context=context, question=question)
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
    # do_sample=False = greedy decoding: same input -> same answer, which is
    # what you want for a reproducible demo and reproducible evaluation.
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0], skip_special_tokens=True).strip()


def main() -> None:
    from src.retriever import retrieve

    q = "Explain tokenization in simple words."
    hits = retrieve(q, top_k=2)
    context = "\n\n".join(h["text"] for h in hits)

    answer = generate(q, context)

    print("Q:", q)
    print(f"context used: {hits[0]['source_file']} #{hits[0]['chunk_index']} "
          f"(score {hits[0]['score']:.3f})")
    print("A:", answer)
    print("\nGenerative check - answer is a verbatim copy of the context:",
          answer in context)


if __name__ == "__main__":
    main()