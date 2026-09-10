Stage 1 decisions (locked)
Recorded so the report's "System Requirements" and "Dataset" sections are written from
decisions we actually made, with reasons — not retro-fitted.

Decision	Choice	Reason
Target hardware	Laptop, no GPU, 16 GB RAM	Rules out large LLMs. Puts a ~250M-parameter encoder-decoder comfortably in range. Makes fast CPU inference a design constraint, not an afterthought.
Knowledge domain	NLP / AI concepts (teaching corpus)	You understand the content deeply, so you can judge answer quality live during the demo — that matters more than it sounds.
Runtime	Local Windows machine (no Colab)	Offline demo = no internet-failure risk during judging. Also makes "fully offline, reproducible, zero API cost" a legitimate selling point.
Consequences of these three decisions
Model size ceiling. We will not consider anything that needs a GPU. Our generator
candidates in Stage 11 will be small instruction-tuned encoder-decoder / small decoder
models, and we will time them on this machine before choosing.
Latency is a rubric item. With CPU-only inference we must target a few seconds per
answer, or the demo drags. This may push us toward the smaller generator + shorter
context, and it is a trade-off we will state openly.
Offline is a feature, not a limitation. We say so in the report.
The domain is "meta" — handle it deliberately. A judge may say "your QA system
answers questions about QA systems?" Our answer: the corpus is a teaching resource and
the domain lets the presenter verify answer correctness in real time; the architecture
is domain-agnostic and the corpus is swappable. If you would rather avoid the question
entirely, say so and we swap the domain in Stage 4 — the architecture does not change.
Corpus sourcing — licensing notes (IMPORTANT, read before Stage 4)
We must only use sources we are legally allowed to ingest. Checked 2026-09-11:

Source	Licence	Verdict for our use
"Natural Language Processing: A Notebook-Based Introduction" — Jie Cao (mlciv.com/nlp-notebooks), v0.1 living edition, OU Alternative Textbook Grant 2025–26	CC BY-NC-SA 4.0	✅ Best primary source. Covers tokenization → embeddings → transformers → fine-tuning → agents, i.e. exactly our topic. NC is fine for a non-commercial college project; SA means we must keep the same licence on derivatives and attribute.
Wikibooks NLP-related books	CC BY-SA 4.0 / GFDL	✅ Usable, permits adaptation; must attribute. Quality varies page to page — we must read what we ingest.
OpenStax, e.g. Principles of Data Science §7.5 "Natural Language Processing"	CC BY-NC-SA, plus an explicit clause: "may not be used in the training of large language models or otherwise be ingested into large language models or generative AI offerings without OpenStax's permission"	⚠️ Do NOT ingest into our RAG corpus. Feeding text to a generative model as context is plausibly "ingesting into a generative AI offering." Reading and citing it is fine; building our index from it is not worth the risk.
Jurafsky & Martin, Speech and Language Processing 3rd-ed. draft	Free to read online; licence not confirmed here	⚠️ Verify on web.stanford.edu yourself before ingesting. Do not assume "free to read" means "free to redistribute/ingest."
Your own written explanations	Yours	✅ Safest option of all, and it lets you control depth and syllabus alignment.
Recommended Stage 4 plan: primary corpus from the CC-licensed notebooks above +
Wikibooks + your own written summaries per topic. Every chunk stores its source and
licence in metadata, so the "Sources" panel in the UI is also a provenance record. That
provenance trail is itself worth a viva point.