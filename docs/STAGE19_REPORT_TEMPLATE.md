Question Answering System using NLP and Generative QA - Report Template
Use YOUR measured numbers, no fabrication. Fill sections with evidence from repo.

1. Abstract (150 words)
Problem: need offline, auditable QA for NLP teaching corpus
Solution: RAG + local FLAN-T5-base + threshold gate
Result: 9-question eval 1.00 accuracy, 0.735 vs 0.108 gap, 3.8s avg latency on i7-1355U CPU-only
2. System Requirements (from STAGE1_DECISIONS.md)
Target: Intel i7-1355U, 16GB RAM, Iris Xe iGPU (CPU-only because torch.xpu unavailable on Iris Xe, verified torch.cuda.is_available() False)
Runtime: Windows, offline, no API key, zero cost
Domain: NLP/AI teaching corpus (9 docs, AI-drafted human-reviewed, disclosure in SOURCES.md)
3. Architecture (ADR-001)
Approach C: RAG + local generative model (chosen over A extractive, B fine-tune, D LLM API because offline + auditable + CPU)
Pipeline: Question -> MiniLM-L6-v2 384-dim -> FAISS IndexFlatIP exact search (65 chunks) -> top_k=3 -> threshold 0.35 gate -> FLAN-T5-base 250M generates 1-2 sentences
Why IndexFlatIP: exact search justified for 65 chunks, no approximation needed
4. Dataset
Corpus: data/documents/ 9 txt files (01-09), 60 paragraphs -> 65 chunks, avg 452 chars, max 599, overlap 100
Provenance: AI-drafted, human-reviewed, SOURCES.md ledger
Processed: data/processed/documents.jsonl, chunks.jsonl
5. Prompt Design (Stage 10b probe - measured at 0.803 context)
A_current: "using only context... answer exactly: insufficient information" -> REFUSED
B_no_escape: "Answer in 1-2 sentences using info above" -> correct but verbatim True
C_soft_escape: "If no relevant info reply I don't know" -> REFUSED "I don't know"
D_own_words (WINNER): "Using context above, answer in your own words in 1-2 sentences" -> correct 0.8028, best trade-off
E_no_copy: longer, still verbatim True
Decision: escape hatch moved to threshold gate (Stage 13), not in prompt
6. Hallucination Control (Stage 13 - measured)
Scores: in-domain 0.779, 0.803, 0.678, 0.830, 0.737, 0.580 avg 0.735
OOD: 0.157 (chocolate cake), 0.108 (IPL), 0.060 (tap) avg 0.108
Gap: ~0.40 vs ~0.16 -> threshold 0.35 protects paraphrase (~0.45) while blocking OOD
Gate: if top_score < 0.35 -> refuse in 0.0s without calling generator, grounded=False, confidence=low
Result: correct_refusal_rate 1.00 (3/3 OOD)
7. Implementation
Embeddings: sentence-transformers all-MiniLM-L6-v2, normalize_embeddings=True, cosine = dot product
Vector store: faiss-cpu 1.15.0 IndexFlatIP, dim 384, 65 vectors
Generator: google/flan-t5-base 990MB safetensors, greedy decode do_sample=False for reproducibility
Core: qa_pipeline.py answer_question() returns {answer, sources, top_score, confidence, grounded, latency}
CLI: python -m src.cli_answer "question" -> JSON (bypass for WDAC)
UI: Flask app_flask.py http://localhost:5000 - UI process never imports torch/pyarrow, child process does
8. Evaluation (Stage 17 - real, n=9, no fabrication)
Test set: 6 in-domain, 3 OOD (list in evaluate.py)
Results:
In-domain avg_score 0.735 grounded_rate 1.00 avg_latency 3.8s (first load 14.1s)
OOD avg_score 0.108 correct_refusal 1.00
Overall accuracy (grounded matches expected) 1.00
Saved: data/processed/eval_results.jsonl
Limitations: n=9 small, needs larger test set for final report; paraphrase scores ~0.45 need protection
9. Baseline Comparison (Stage 18)
TF-IDF baseline: Question -> TF-IDF cosine -> stored chunk verbatim
Results:
Tokenization: TF-IDF 0.203 vs dense 0.803
RAG: TF-IDF 0.405 vs dense 0.678
OOD cake: TF-IDF 0.165 returns irrelevant chunk (hallucinates), RAG correctly refuses 0.157
Conclusion: dense embeddings better semantic match, generative answer rewrites vs verbatim copy, threshold prevents OOD hallucination
10. Demo (Stage 15)
Flask UI screenshot: explain tokenization -> 0.828 high grounded True 16.9s, sources [0.828] 02_tokenization.txt #0 etc.
Streamlit attempted but blocked by WDAC: torch._C and pyarrow.lib DLLs blocked by Application Control policy after Streamlit install (which pulls pyarrow)
Fix: uninstall pyarrow, downgrade sklearn 1.3.2 + numpy 1.26.4, Flask subprocess bypass
This fix is viva point: engineering under constraints
11. Limitations & Future Work
Verbatim copy True for definition questions (context is definition sentence) - still generative token-by-token but could improve paraphrase with larger model
Latency 3.8s avg CPU, first load 14-16s - need model pre-warm for demo
Test set n=9 small - expand to 30-50 for final evaluation
FLAN-T5-base 250M vs small 80M trade-off not benchmarked due to WDAC, future Stage 11 model comparison
12. Conclusion
End-to-end RAG works offline on ordinary laptop, auditable sources+scores, measured threshold prevents hallucination, real evaluation 1.00 accuracy on 9-question set
Appendix: How to Run
text

python -m venv venv
.\venv\Scripts\activate
python -m pip install -r requirements.txt  # use python -c write_bytes method for requirements.txt on PowerShell
python -m src.preprocessing
python -m src.chunking
python -m src.embeddings
python -m src.vector_store
python -m src.retriever
python -m src.generator  # should give real answer at 0.803
python -m src.qa_pipeline
python -m src.cli_answer "Explain tokenization..."
python -m src.evaluate  # 9 questions
python app_flask.py  # http://localhost:5000
Appendix: Git History
Incremental commits: Stage 5 preprocessing, Stage 6 chunking 60->65, Stage 7 embeddings 384-dim 0.521 vs 0.033, Stage 8 FAISS 65x384, Stage 9 retriever 0.779/0.454/0.157, Stage 10 generator 0.803 + prompt probe, Stage 13-14 qa_pipeline, Stage 15 Flask UI 0.828, Stage 16-17 eval 1.00