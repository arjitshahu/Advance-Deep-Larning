"""Shared settings for the College Knowledge Assistant."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
CHUNKS_PATH = VECTORSTORE_DIR / "chunks.json"
INDEX_PATH = VECTORSTORE_DIR / "index.faiss"

# Small, general-purpose multilingual sentence embedding model.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Ollama runs a local language model. Pull the model once with `ollama pull`.
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:3b"

CHUNK_SIZE = 850
CHUNK_OVERLAP = 140
TOP_K = 5
# This is a heuristic, not a calibrated probability. Tune using your own test set.
MIN_SIMILARITY = 0.28
