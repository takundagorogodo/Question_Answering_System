"""
Central configuration for the Generative QA project.

Every path and model name lives HERE so that if anything moves, we change
one file. All other modules import from here.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---- data folders ----------------------------------------------------------
CORPUS_DIR    = PROJECT_ROOT / "data" / "documents"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INDEX_DIR     = PROJECT_ROOT / "data" / "index"

# ---- other folders ----------------------------------------------------------
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR   = PROJECT_ROOT / "logs"

# ---- models ----------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATOR_MODEL = "google/flan-t5-base"  # Stage 10 winner, measured 0.803 context

# ---- retrieval knobs (measured, not guessed) -------------------------------
# From your probe:
# in-domain: 0.803 (tokenization), 0.678 (RAG), 0.830 (hallucination)
# OOD: 0.157 (chocolate cake)
# Gap 0.40 vs 0.16 -> threshold 0.35 protects paraphrase (~0.45) while blocking OOD
RETRIEVAL_TOP_K = 3
SIMILARITY_THRESHOLD = 0.35

# ---- chunking --------------------------------------------------------------
CHUNK_MAX_CHARS = 500
CHUNK_OVERLAP_CHARS = 100

