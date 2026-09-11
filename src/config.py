
"""
Central configuration for the Generative QA project.

Every path and model name lives HERE so that if anything moves, we change
one file. All other modules import from here.
"""
from pathlib import Path

# Project root = the folder that CONTAINS src/ (one level above this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---- data folders ----------------------------------------------------------
CORPUS_DIR    = PROJECT_ROOT / "data" / "documents"   # raw corpus (Stage 4)
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"   # cleaned JSONL (Stage 5)
INDEX_DIR     = PROJECT_ROOT / "data" / "index"       # FAISS index (Stage 8)

# ---- other folders ----------------------------------------------------------
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR   = PROJECT_ROOT / "logs"

# ---- models (used from Stage 7 / 10 onward) --------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # ~23M params, 384-dim
GENERATOR_MODEL = "google/flan-t5-base"  # C1 candidate; benchmarked vs C2 in Stage 11

# ---- retrieval knobs (Stages 9/13 - tune with measurements, not guesses) ----
RETRIEVAL_TOP_K = 3          # how many chunks the generator receives
SIMILARITY_THRESHOLD = 0.35  # below this -> "insufficient information" (Stage 13)

# ---- chunking (Stage 6) ------------------------------------------------------
CHUNK_MAX_CHARS = 500     # target chunk size in characters
CHUNK_OVERLAP_CHARS = 100 # overlap so a cut never loses context mid-thought