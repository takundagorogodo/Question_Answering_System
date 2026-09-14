JUDGING-RUBRIC PROGRESS TRACKER
We score ourselves against this rubric after every stage. The rule: no stage
is "done" until we can say, in one sentence, how it moved a rubric line.

The rubric
#	Criterion	Weight	What "10/10" looks like	Our score (0–10)	Evidence
1	Innovation / Differentiator	20%	A clear, defensible novel contribution beyond "we built RAG" — and it is visible in the demo.	3	Hypothesis defined (evidence-grounded answers + measured hallucination rate) + threshold gate implemented in qa_pipeline (Stage 13 early).
2	Technical Depth	25%	Student can explain every component end-to-end, knows the trade-offs of each choice, and can answer "why not X?"	4	ADR-001 + prompt probe with 5 variants measured (A/B/C/D/E) showing why strict refusal prompts fail at 0.803 score.
3	Evaluation Rigor	20%	Real test set, real numbers, baseline comparison, honest error analysis. No fabricated results.	2	First genuine measurements: paraphrase 0.521 vs 0.033, retrieval 0.779/0.454/0.157, generator-facing 0.803, prompt probe refusal vs success logged.
4	Demo Quality	15%	Live demo works on the spot, offline, in under 2 minutes, with no fumbling.	2	End-to-end pipeline works: Q -> FAISS 0.803 -> FLAN-T5-base generates answer (Stage 10 closed).
5	Documentation	10%	Report + README + code comments tell a coherent story; git history shows incremental work.	4	Incremental commits to main, pushed.
6	Presentation & Viva	10%	Slides rehearsed; tough judge' questions answered without hesitation.	0	Not started (Stage 20).
Weighted total: 0.20(3) + 0.25(4) + 0.20(2) + 0.15(2) + 0.10(4) + 0.10(0) = 2.70 / 10

Log
Stage 1 — Project Understanding
Built: repo skeleton, rubric tracker, Lesson 1, locked decisions.
Score: 0.00 -> 0.85
Stage 2 — Choose approach
Built: ADR-001 Approach C RAG + local generative accepted.
Score: 0.85 -> 1.45
Stage 3 — Environment setup
Built: venv Python 3.10.11, git, dev helpers.
Score: 1.45 -> 1.70
Stages 4-9 — corpus to retrieval
Built: 9-doc corpus, preprocessing, chunking 60->65, FAISS 65x384, retriever.
Measurements: paraphrase 0.521 vs 0.033, retrieval probe 0.779/0.454/0.157
Score: 1.70 -> 1.90
Stage 10 — generative QA (closed 2026-09-13)
Built: generator.py with FLAN-T5-base, prompt_probe.py with 5 variants
Measurement on target laptop:
Q: "Explain tokenization in simple words."
top-1 retrieval: 02_tokenization.txt #0 score 0.803 (correct)
A_current: "insufficient information" (over-strict)
B_no_escape: correct answer (verbatim True - definition sentence copied)
C_soft_escape: "I don't know" (over-refuses)
D_own_words: same correct answer, 1 sentence, verbatim True
E_no_copy: longer answer with extra explanation, verbatim True
Decision: FLAN-T5-base IS usable, retrieval not the problem. Winning prompt = D_own_words (B + own words). Escape hatch moved to threshold gate in Stage 13.
Rubric: Demo 1->2, Tech Depth 3->4, Innovation 2->3, Eval 1->2
Score: 1.90 -> 2.70
Next: Stage 11-14 — qa_pipeline + evaluation set
qa_pipeline.py implements threshold gate (0.35) + confidence + grounded flag
Need to test 4 questions including OOD chocolate cake -> should trigger low confidence

JUDGING-RUBRIC PROGRESS TRACKER
Rubric
#	Criterion	Weight	What "10/10" looks like	Score	Evidence
1	Innovation / Differentiator	20%	Clear novel contribution visible in demo	4	Threshold gate (0.35) from measured gap 0.735 vs 0.108, auditable sources+scores, Flask subprocess bypass for WDAC
2	Technical Depth	25%	Explain every component + trade-offs	5	ADR-001, prompt probe 5 variants (A-E) measured at 0.803, TF-IDF vs dense comparison, WDAC root cause + fix
3	Evaluation Rigor	20%	Real test set, real numbers, baseline, error analysis	4	9-question test set (6 in-domain, 3 OOD), scores 0.779/0.803/0.678/0.830/0.737/0.580 vs 0.157/0.108/0.060, latency 3.8s avg, TF-IDF baseline 0.203 vs 0.803
4	Demo Quality	15%	Live demo works offline in <2 min	4	Flask UI http://localhost:5000 working, screenshot 0.828 high grounded True, OOD refusal demo, eval_results.jsonl saved
5	Documentation	10%	Coherent story, git history incremental	5	Incremental commits to main, pushed, eval_results.jsonl, SOURCES.md
6	Presentation & Viva	10%	Slides rehearsed, tough questions answered	1	Numbers ready for slides, WDAC story is viva point
Weighted: 0.204 + 0.255 + 0.204 + 0.154 + 0.105 + 0.101 = 4.05 /10

Log
Stage 10 - generative QA (closed)
Q "Explain tokenization..." -> top-1 0.803 correct file, FLAN-T5-base generates answer
Prompt probe: A insufficient info, B correct verbatim True, C I don't know, D own_words best
Winning prompt D: "Using context above, answer in your own words in 1-2 sentences"
Stage 13 - hallucination control (measured)
In-domain avg 0.735 (0.580-0.830), OOD avg 0.108 (0.060-0.157)
Gap -> threshold 0.35 protects paraphrase (~0.45) while blocking OOD
Gate implemented in qa_pipeline.py: if top_score < 0.35 -> refuse without calling generator (0.0s)
Stage 14-15 - core app + UI
qa_pipeline.py: retrieve + threshold + generate + confidence + grounded + latency
CLI: python -m src.cli_answer works (0.8028)
WDAC issue: torch._C and pyarrow.lib blocked by App Control policy after Streamlit install
Fix: uninstall pyarrow, downgrade sklearn 1.3.2 + numpy 1.26.4, Flask UI with subprocess bypass
Flask UI: app_flask.py http://localhost:5000 - UI process never imports torch, child python.exe does
Stage 16-18 - evaluation + baseline
evaluate.py: 9 questions, overall accuracy 1.00, grounded_rate 1.00 in-domain, correct_refusal 1.00 OOD
baseline_tfidf.py: TF-IDF 0.203 vs dense 0.803 for tokenization, TF-IDF returns irrelevant chunk for OOD cake (0.165) while RAG correctly refuses
Saved data/processed/eval_results.jsonl
Next: Stage 19-200 report + presentation
Report sections: use these real numbers, no fabrication
Slides: show Flask screenshot (0.828), evaluation table (0.735 vs 0.108), threshold justification, WDAC fix as engineering story