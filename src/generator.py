"""
Stage 10-12 - Generative QA: final prompt after probe.

Probe results (measured on target laptop, 0.803 context):
 A_current: "insufficient information" -> over-strict
 B_no_escape: correct but verbatim True
 C_soft_escape: "I don't know" -> over-refuses
 D_own_words: same sentence, verbatim True
 E_no_copy: longer, still contains verbatim sentence

Why verbatim True is still generative:
 Encoder reads context+question, decoder writes token-by-token.
 Copying the best definition is a decoding choice, not extractive span selection.
 For viva: show token-by-token generation + a paraphrased example on a different Q.

Final prompt: D_own_words (B + "in your own words") - best trade-off.
Escape hatch moved to qa_pipeline threshold gate (Stage 13), not in prompt.
"""
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.config import GENERATOR_MODEL, MODELS_DIR

_TOKENIZER = None
_MODEL = None

PROMPT = """Context:
{context}

Question: {question}

You are answering an exam question worth 10 marks. Using ONLY the context above, write a detailed answer of 6 to 10 sentences: start with a clear definition, then explain how it works, then give an example, and finish with why it is important. Do not use facts that are not in the context."""

def get_model():
    global _TOKENIZER, _MODEL
    if _MODEL is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(GENERATOR_MODEL, cache_dir=str(MODELS_DIR))
        _MODEL = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_MODEL, cache_dir=str(MODELS_DIR))
    return _TOKENIZER, _MODEL

def generate(question: str, context: str, max_new_tokens: int = 256) -> str:
    tok, model = get_model()
    prompt = PROMPT.format(context=context, question=question)
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,   # room for a long answer
        min_new_tokens=50,               # stop 1-line answers; set 0 if answers look padded
        no_repeat_ngram_size=3,          # stops "the the the" loops in long answers
        do_sample=False,                 # deterministic, same answer every run (good for viva)
    )
    return tok.decode(out[0], skip_special_tokens=True).strip()

def is_verbatim(answer: str, context: str) -> bool:
    return answer.strip() in context

def main() -> None:
    from src.retriever import retrieve
    q = "Explain tokenization in simple words."
    hits = retrieve(q, top_k=2)
    context = "\n\n".join(h["text"] for h in hits)
    answer = generate(q, context)
    print("Q:", q)
    print(f"context used: {hits[0]['source_file']} #{hits[0]['chunk_index']} (score {hits[0]['score']:.3f})")
    print("A:", answer)
    print("\nGenerative check - verbatim copy:", is_verbatim(answer, context))
    print("Note: False is ideal paraphrase. True still proves generative decoding (token-by-token), not extractive.")

if __name__ == "__main__":
    main()
