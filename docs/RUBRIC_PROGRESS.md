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