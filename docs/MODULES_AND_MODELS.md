Modules, Models, and NLP Concepts Used (viva reference)
Modules (pipeline order)
src/config.py — central settings: model names, chunk size (500), overlap (100), top-K (3), threshold (0.35). One file controls everything.
src/preprocessing.py — cleans the raw .txt corpus: regex whitespace removal, lowercasing (normalization).
src/chunking.py — splits each document into 500-character chunks with 100-char overlap → 65 chunks; each chunk keeps source file + index for traceability.
src/vector_store.py — embeds all chunks with MiniLM (384-dim) and saves faiss.index + chunks.jsonl. Run once at build time.
src/retriever.py — embeds the user question, searches the FAISS index with cosine similarity, returns top-3 chunks with scores.
src/generator.py — FLAN-T5-base reads question + retrieved chunks and writes a detailed 10-mark answer (deterministic, no sampling).
src/qa_pipeline.py — the glue: retrieve → confidence gate (score < 0.35 = refuse "insufficient information") → generate → attach confidence + grounded flag + sources.
src/cli_answer.py / app_flask.py — two front-ends, one pipeline: CLI (JSON output) and Flask web UI (shows answer, score, grounded flag, sources).
src/evaluate.py — 9-question benchmark: 6 in-domain, 3 out-of-domain; measures scores, grounded rate, refusal rate, latency.
src/baseline_tfidf.py — TF-IDF baseline for comparison only (0.203 vs 0.803 on the same question).
Models used
sentence-transformers/all-MiniLM-L6-v2 — embedding model, 22M params (~80 MB). Turns any text into one 384-dim vector; similar meaning = high cosine. Runs fast on CPU.
google/flan-t5-base — generative model, encoder-decoder transformer (~250M params, ~990 MB). Instruction-tuned, so it follows prompts like "answer using only the context". The ONLY model that writes text.
FAISS IndexFlatIP — not a model; a vector index library. Exact inner-product (cosine) search over the 65 chunk vectors.
TF-IDF (scikit-learn) — baseline vectorizer only; word counts, no meaning. Kept to prove dense embeddings are better.
NLTK concepts (classic NLP = same ideas, modern tools)
Corpus — our 9 .txt study documents (a corpus is any organized text collection).
Tokenization — splitting text into units. NOT done by NLTK: both our models contain their own subword tokenizer (SentencePiece-style) that splits rare words into pieces.
Normalization / cleaning — lowercasing + whitespace/punctuation cleanup, done with regex in preprocessing.py.
Stop words — known concept, deliberately NOT removed: transformer models need full sentences for context (we explain this trade-off in viva).
Stemming / lemmatization — known concept, deliberately NOT applied before embedding; MiniLM's subword tokenizer handles word variants internally.
Cosine similarity — the retrieval math (1 = same meaning, 0 = unrelated; measured 0.521 paraphrase vs 0.033 unrelated).
N-grams — concept appears in generation as no_repeat_ngram_size=3 (blocks 3-word repetition loops in long answers).
NLTK library itself — not imported anywhere; its classic concepts are implemented via regex + Hugging Face tokenizers, which we state honestly in the report.