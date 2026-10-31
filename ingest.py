"""Read PDF/TXT files, chunk their text, create embeddings, and save a FAISS index."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import faiss
import numpy as np
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

from config import (
    CHUNK_OVERLAP, CHUNK_SIZE, DOCUMENTS_DIR, EMBEDDING_MODEL,
    INDEX_PATH, VECTORSTORE_DIR, CHUNKS_PATH,
)


def read_document(path: Path) -> list[dict]:
    """Return one record per PDF page, or one record for a TXT file."""
    records = []
    if path.suffix.lower() == ".pdf":
        with fitz.open(path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()
                if text:
                    records.append({
                        "source": path.name, "page": page_number, "text": text
                    })
    elif path.suffix.lower() in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            records.append({"source": path.name, "page": None, "text": text})
    return records


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Simple character-based chunking with overlap to preserve boundary context."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0.")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Prefer ending at a nearby word boundary without making tiny chunks.
        if end < len(text):
            boundary = text.rfind(" ", start + int(chunk_size * 0.65), end)
            if boundary > start:
                end = boundary
        part = text[start:end].strip()
        if part:
            chunks.append(part)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_knowledge_base(
    documents_dir: Path = DOCUMENTS_DIR,
    vectorstore_dir: Path = VECTORSTORE_DIR,
    embedding_model: str = EMBEDDING_MODEL,
) -> int:
    documents_dir = Path(documents_dir)
    vectorstore_dir = Path(vectorstore_dir)
    supported = {".pdf", ".txt", ".md"}
    paths = sorted(
        p for p in documents_dir.iterdir()
        if p.is_file() and p.suffix.lower() in supported
    ) if documents_dir.exists() else []
    if len(paths) < 5:
        raise ValueError(
            f"Add at least 5 PDF/TXT/MD documents to {documents_dir}; found {len(paths)}."
        )

    chunks = []
    for path in paths:
        for record in read_document(path):
            for chunk_number, part in enumerate(split_text(record["text"]), start=1):
                chunks.append({
                    "source": record["source"],
                    "page": record["page"],
                    "chunk": chunk_number,
                    "text": part,
                })
    if not chunks:
        raise ValueError("No readable text found. For scanned PDFs, run OCR first.")

    print(f"Embedding {len(chunks)} chunks from {len(paths)} documents...")
    model = SentenceTransformer(embedding_model)
    vectors = model.encode(
        [item["text"] for item in chunks],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])  # cosine similarity after normalization
    index.add(vectors)

    vectorstore_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(vectorstore_dir / "index.faiss"))
    (vectorstore_dir / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Done. Saved {len(chunks)} chunks to {vectorstore_dir.resolve()}")
    return len(chunks)


if __name__ == "__main__":
    try:
        build_knowledge_base()
    except Exception as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
