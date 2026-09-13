"""
Stage 10 - Generative QA: (question + retrieved context) -> WRITTEN answer.

WHY this is "generative":
  The model does NOT return a stored answer. It reads context + question
  and decodes a new sentence token by token. Check at bottom must be
  False ideally (not a verbatim copy).

FIX from prompt_probe (Stage 10b):
  A_current -> "insufficient information" (over-strict, score was 0.803)
  B_no_escape -> correct answer, but verbatim copy True
  C_soft_escape -> "I don't know" (over-refuses)
  -> Winner is B. We keep B and add "in your own words" to push paraphrase.
  -> Escape hatch ("insufficient info") moves to Stage 13 with threshold gate,
     NOT inside the generator prompt for Stage 10.
"""
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.config import GENERATOR_MODEL, MODELS_DIR

_TOKENIZER = None
_MODEL = None

# Winning prompt from probe: B + "in your own words" to reduce verbatim copy
PROMPT = """Context:
{context}

Question: {question}

Using the context above, answer the question in your own words in one or two sentences."""

def get_model():
    global _TOKENIZER, _MODEL
    if _MODEL is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(GENERATOR_MODEL, cache_dir=str(MODELS_DIR))
        _MODEL = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_MODEL, cache_dir=str(MODELS_DIR))
    return _TOKENIZER, _MODEL

def generate(question: str, context: str, max_new_tokens: int = 128) -> str:
    tok, model = get_model()
    prompt = PROMPT.format(context=context, question=question)
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0], skip_special_tokens=True).strip()

def main() -> None:
    from src.retriever import retrieve
    q = "Explain tokenization in simple words."
    hits = retrieve(q, top_k=2)
    context = "\n\n".join(h["text"] for h in hits)
    answer = generate(q, context)
    print("Q:", q)
    print(f"context used: {hits[0]['source_file']} #{hits[0]['chunk_index']} (score {hits[0]['score']:.3f})")
    print("A:", answer)
    print("\nGenerative check - answer is a verbatim copy of the context:", answer in context)
    print("If False -> good generative proof. If True -> still generative (token-by-token decode), but try to paraphrase more.")

if __name__ == "__main__":
    main()
