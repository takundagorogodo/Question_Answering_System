from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-base"

# Load once at import time - loading is slow, so we don't want to repeat it per question
_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


PROMPT_TEMPLATE = """Context:
{context}

Question:
{question}

Instructions:
Answer the question using only the provided context.
Give a complete, explanatory answer in at least one full sentence, not a single word.
If the answer cannot be found in the context, say that sufficient information is unavailable.
"""

def build_prompt(question, retrieved_chunks):
    """
    Combine retrieved chunk texts into a single context block,
    then insert into our prompt template alongside the question.
    """
    context = "\n\n".join(chunk["text"] for chunk in retrieved_chunks)
    return PROMPT_TEMPLATE.format(context=context, question=question)

def generate_answer(question, retrieved_chunks, max_length=150):
    """
    Build the prompt, run it through FLAN-T5-base, and return the generated answer.
    """
    prompt = build_prompt(question, retrieved_chunks)

    inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = _model.generate(
        **inputs,
        max_length=max_length,
        min_length=30,           # forces the model to keep generating past a single word
        num_beams=4,             # beam search instead of greedy decoding — better quality
        no_repeat_ngram_size=3,  # avoids repetitive loops
        early_stopping=True
    )

    answer = _tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

if __name__ == "__main__":
    from src.preprocessing import load_documents
    from src.chunking import chunk_documents
    from src.retrieval import VectorStore

    docs = load_documents()
    chunks = chunk_documents(docs)

    store = VectorStore()
    store.build(chunks)

    question = "What is the difference between extractive and generative QA?"
    retrieved = store.search(question, top_k=3)

    answer = generate_answer(question, retrieved)

    print(f"Question: {question}\n")
    print("Retrieved context from:")
    for r in retrieved:
        print(f"  - {r['title']} ({r['chunk_id']}), score={r['score']:.4f}")
    print(f"\nGenerated Answer:\n{answer}")