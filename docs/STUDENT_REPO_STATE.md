Student repository state (observed from screenshot, 2026-09-11)
Facts visible in the VS Code screenshot — recorded so we don't re-derive or misremember them.
Anything NOT listed here is unknown, including the contents of the files.

Observed
Project folder: C:\Users\Prosperity\Documents\Question_Answering_System
Git repo already existed before this stage: git status shows On branch main,
Your branch is ahead of 'origin/main' by 2 commits → an origin remote exists
(likely GitHub) and at least two local commits are unpushed.
git version 2.55.0.windows.3 → git is installed and on PATH.
Working tree was clean at screenshot time → everything visible in the explorer is
either committed or ignored.
Folder structure already present, and it differs from my skeleton:
mine has: baseline, cache, data, docs, evaluation, logs, models, notebooks, src
theirs ALSO has: embeddings/ (top level), tests/
src/ already contains files matching my planned module names:
app.py, chunking.py, config.py, embeddings.py, generator.py, grounding.py, pipeline.py, preprocessing.py, prompt_builder.py, retriever.py, vector_store.py
plus a __pycache__/ (→ some code was executed at some point).
Contents of those files: UNKNOWN. They may be empty stubs or a previous attempt.
Must be reviewed before Stages 5–15 write to them. Do not silently overwrite.
Root files: .gitignore, dev.ps1, README.md, requirements.txt, STAGE1_DECISIONS.md
(theirs at root, mine at docs/; cosmetic mismatch only) and setuo_project.ps1
(a typo of setup_project.ps1).
setup_project.ps1 was run under the misspelled name and failed with
CommandNotFoundException → at the moment it was invoked, no file with that exact
name existed (file saved/renamed afterwards).
venv: healthy now (pyvenv.cfg True, pip 26.2.1, Python 3.10.11, prefix (venv)).
Unknown (must ask before assuming)
Where the src/*.py files and embeddings/, tests/ came from (previous attempt?
class template? another tool?) and whether they contain code.
What the two unpushed commits contain.
Contents of their requirements.txt and README.md.
Consequences
git init / identity steps are skipped — repo already live.
setup_project.ps1 must be renamed to the correct spelling and re-run (idempotent) so
.gitignore is the tested version and the remaining folders exist.
Before any module is written, review existing src/*.py — per teaching rule: review,
never silently replace.