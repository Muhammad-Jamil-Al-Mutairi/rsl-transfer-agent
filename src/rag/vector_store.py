"""
Hybrid RAG vector store for the RSL Transfer Market & FFP Advisor Agent.

Pipeline:
  1. Load .txt dossiers from data/sample_documents/.
  2. Split on "=== Page N: Title ===" markers so every chunk keeps an exact
     source document name + page number for citation purposes.
  3. Further split long pages into overlapping character-window chunks.
  4. Embed each chunk with Gemini's `gemini-embedding-001` model, truncated
     to 768 dimensions via `output_dimensionality` (dense vectors), and also
     cache the raw chunk text locally for BM25 (sparse keyword) search.
  5. Upsert the dense vectors into a Qdrant Cloud collection with
     source/page metadata.
  6. `search_documents` performs TRUE hybrid search: it runs a Qdrant dense
     vector search and a local BM25 keyword search in parallel, then fuses
     both rankings with Reciprocal Rank Fusion (RRF) before returning the
     top_k chunks with citations and scores.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

load_dotenv()

EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
# gemini-embedding-001 defaults to 3072-dim (Matryoshka) output; we request
# 768 explicitly via output_dimensionality in _embed_texts() below so it
# matches the Qdrant collection's vector size.
EMBEDDING_DIM = 768
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "rsl_transfer_dossiers")
DEFAULT_DOC_DIR = Path(__file__).resolve().parents[2] / "data" / "sample_documents"

CHUNK_SIZE_CHARS = 900
CHUNK_OVERLAP_CHARS = 150

_PAGE_HEADER_RE = re.compile(r"===\s*Page\s+(\d+)\s*:\s*(.*?)\s*===", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Local cache of indexed chunks used to run BM25 keyword search alongside
# Qdrant's dense vector search (see "Hybrid keyword search (BM25)" section).
_BM25_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / ".bm25_chunk_cache.json"
RRF_K = 60  # standard Reciprocal Rank Fusion smoothing constant


@dataclass
class DocChunk:
    text: str
    source: str
    page: int
    page_title: str
    chunk_index: int


# ---------------------------------------------------------------------------
# Client helpers
# ---------------------------------------------------------------------------

def _get_gemini_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in the environment.")
    return genai.Client(api_key=api_key)


def get_qdrant_client() -> QdrantClient:
    url = os.environ.get("QDRANT_URL")
    api_key = os.environ.get("QDRANT_API_KEY")
    if not url:
        raise RuntimeError("QDRANT_URL is not set in the environment.")
    return QdrantClient(url=url, api_key=api_key)


def check_gemini_status() -> dict[str, Any]:
    """Lightweight, no-network-call check that GEMINI_API_KEY is configured."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {"connected": False, "error": "GEMINI_API_KEY not set."}
    try:
        _get_gemini_client()
        return {"connected": True}
    except Exception as exc:  # pragma: no cover - defensive
        return {"connected": False, "error": str(exc)}


def check_qdrant_status() -> dict[str, Any]:
    """Verifies Qdrant Cloud connectivity and reports current collection size."""
    try:
        qdrant = get_qdrant_client()
        collections = [c.name for c in qdrant.get_collections().collections]
        collection_exists = COLLECTION_NAME in collections
        points_count = None
        if collection_exists:
            points_count = qdrant.count(collection_name=COLLECTION_NAME, exact=True).count
        return {
            "connected": True,
            "collection_exists": collection_exists,
            "collection": COLLECTION_NAME,
            "points_count": points_count,
        }
    except Exception as exc:
        return {"connected": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _split_into_pages(full_text: str, source_name: str) -> list[tuple[int, str, str]]:
    """Split raw dossier text into (page_number, page_title, page_text) tuples."""
    matches = list(_PAGE_HEADER_RE.finditer(full_text))
    if not matches:
        return [(1, source_name, full_text)]

    pages: list[tuple[int, str, str]] = []
    for i, match in enumerate(matches):
        page_num = int(match.group(1))
        title = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        page_text = full_text[start:end].strip()
        if page_text:
            pages.append((page_num, title, page_text))
    return pages


def _chunk_text(text: str, size: int = CHUNK_SIZE_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    """Sliding-window chunking that prefers breaking on paragraph/sentence boundaries."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind("\n\n", start, end)
            if boundary == -1 or boundary <= start + size // 2:
                boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start + size // 2:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def load_and_chunk_documents(doc_dir: Path | str = DEFAULT_DOC_DIR) -> list[DocChunk]:
    """Load every .txt file in doc_dir and return page/citation-aware chunks."""
    doc_dir = Path(doc_dir)
    all_chunks: list[DocChunk] = []
    for file_path in sorted(doc_dir.glob("*.txt")):
        raw_text = file_path.read_text(encoding="utf-8")
        for page_num, title, page_text in _split_into_pages(raw_text, file_path.name):
            for idx, chunk_text_ in enumerate(_chunk_text(page_text)):
                all_chunks.append(
                    DocChunk(
                        text=chunk_text_,
                        source=file_path.name,
                        page=page_num,
                        page_title=title,
                        chunk_index=idx,
                    )
                )
    return all_chunks


# ---------------------------------------------------------------------------
# Hybrid keyword search (BM25) cache
# ---------------------------------------------------------------------------

_bm25_cache_memo: list[DocChunk] | None = None


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer used for BM25 indexing/querying."""
    return _TOKEN_RE.findall(text.lower())


def _save_bm25_cache(chunks: list[DocChunk]) -> None:
    """Persist the indexed chunks to a local JSON cache so BM25 can run against
    the exact same corpus that was just embedded into Qdrant. Best-effort:
    indexing must still succeed even if this local cache write fails."""
    global _bm25_cache_memo
    try:
        _BM25_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "text": c.text,
                "source": c.source,
                "page": c.page,
                "page_title": c.page_title,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ]
        _BM25_CACHE_PATH.write_text(json.dumps(payload), encoding="utf-8")
        _bm25_cache_memo = list(chunks)
    except OSError:
        pass


def _load_bm25_cache() -> list[DocChunk] | None:
    """Load the local BM25 chunk cache, memoized in-process. Returns None if
    the cache is missing or unreadable so callers can fall back gracefully."""
    global _bm25_cache_memo
    if _bm25_cache_memo is not None:
        return _bm25_cache_memo
    if not _BM25_CACHE_PATH.exists():
        return None
    try:
        raw = json.loads(_BM25_CACHE_PATH.read_text(encoding="utf-8"))
        chunks = [DocChunk(**item) for item in raw]
    except (OSError, json.JSONDecodeError, TypeError, KeyError):
        return None
    _bm25_cache_memo = chunks
    return chunks


def _bm25_search(query: str, candidate_pool: int) -> list[tuple[tuple, dict[str, Any]]]:
    """Rank cached chunks against `query` with BM25Okapi.

    Returns a list of (key, payload) tuples ordered best-first, where key is
    the (source, page, chunk_index) tuple used to align sparse hits with
    dense Qdrant hits during fusion. Returns [] if the BM25 cache or the
    `rank_bm25` dependency is unavailable, so hybrid search safely degrades
    to dense-only search instead of crashing.
    """
    chunks = _load_bm25_cache()
    if not chunks:
        return []

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return []

    tokenized_corpus = [_tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(_tokenize(query))

    ranked_indices = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)

    ranked: list[tuple[tuple, dict[str, Any]]] = []
    for i in ranked_indices[:candidate_pool]:
        if scores[i] <= 0:
            continue  # no keyword overlap at all; don't let it dilute the fusion
        chunk = chunks[i]
        key = (chunk.source, chunk.page, chunk.chunk_index)
        ranked.append(
            (
                key,
                {
                    "text": chunk.text,
                    "source": chunk.source,
                    "page": chunk.page,
                    "page_title": chunk.page_title,
                    "sparse_score": round(float(scores[i]), 4),
                },
            )
        )
    return ranked


def _reciprocal_rank_fusion(
    dense_ranked: list[tuple[tuple, dict[str, Any]]],
    sparse_ranked: list[tuple[tuple, dict[str, Any]]],
    k: int = RRF_K,
) -> list[dict[str, Any]]:
    """Fuse dense-vector and BM25 rankings with Reciprocal Rank Fusion:
    RRF_score(doc) = 1 / (k + dense_rank) + 1 / (k + sparse_rank), where a
    chunk missing from one of the two rankings simply doesn't get that term.
    """
    rrf_scores: dict[tuple, float] = {}
    payload_by_key: dict[tuple, dict[str, Any]] = {}

    for rank, (key, payload) in enumerate(dense_ranked, start=1):
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
        payload_by_key[key] = payload

    for rank, (key, payload) in enumerate(sparse_ranked, start=1):
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
        payload_by_key.setdefault(key, payload)

    fused = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)
    return [{"rrf_score": score, **payload_by_key[key]} for key, score in fused]


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _embed_texts(client: genai.Client, texts: list[str], task_type: str) -> list[list[float]]:
    if not texts:
        return []
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBEDDING_DIM),
    )
    return [list(embedding.values) for embedding in result.embeddings]


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def _recreate_collection(qdrant: QdrantClient) -> None:
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME in existing:
        qdrant.delete_collection(COLLECTION_NAME)
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=qmodels.VectorParams(size=EMBEDDING_DIM, distance=qmodels.Distance.COSINE),
    )


def index_documents(doc_dir: Path | str = DEFAULT_DOC_DIR, batch_size: int = 16) -> dict[str, Any]:
    """Chunk, embed, and upsert every .txt dossier in doc_dir into Qdrant Cloud.

    This recreates the collection from scratch each call, so re-indexing is
    always idempotent (safe to trigger repeatedly from the "Re-index
    Transfer Dossier" sidebar button).

    Returns:
        A dict describing how many chunks were indexed, or an error.
    """
    try:
        gemini = _get_gemini_client()
        qdrant = get_qdrant_client()
    except RuntimeError as exc:
        return {"success": False, "error": str(exc)}

    chunks = load_and_chunk_documents(doc_dir)
    if not chunks:
        return {"success": False, "error": f"No .txt documents found in {doc_dir}"}

    try:
        _recreate_collection(qdrant)

        total_upserted = 0
        sources = set()
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            vectors = _embed_texts(gemini, [c.text for c in batch], task_type="RETRIEVAL_DOCUMENT")
            points = [
                qmodels.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "text": chunk.text,
                        "source": chunk.source,
                        "page": chunk.page,
                        "page_title": chunk.page_title,
                        "chunk_index": chunk.chunk_index,
                    },
                )
                for chunk, vector in zip(batch, vectors)
            ]
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            total_upserted += len(points)
            sources.update(c.source for c in batch)

        # Cache the same chunks locally so search_documents can run BM25
        # keyword search over the exact corpus that was just embedded.
        _save_bm25_cache(chunks)
    except Exception as exc:
        return {"success": False, "error": str(exc)}

    return {
        "success": True,
        "chunks_indexed": total_upserted,
        "documents_indexed": sorted(sources),
        "collection": COLLECTION_NAME,
        "source_dir": str(doc_dir),
    }


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def search_documents(query: str, top_k: int = 4) -> dict[str, Any]:
    """Hybrid search: fuse Qdrant dense-vector search with local BM25 keyword
    search via Reciprocal Rank Fusion (RRF), then return the top_k chunks.

    Args:
        query: Natural-language search query.
        top_k: Number of chunks to retrieve.

    Returns:
        A dict with a list of results, each carrying `text`, `source`,
        `page`, `page_title`, and a `score` (the fused RRF score when the
        BM25 cache is available, otherwise the raw dense cosine similarity),
        ready to be rendered as citations. Shape is always
        {"success": True, "results": [...], ...} on success, matching what
        `src/agent/orchestrator.py` expects.
    """
    if not query or not query.strip():
        return {"success": False, "error": "query must be non-empty."}

    # Fetch a wider candidate pool than top_k from each retrieval method so
    # RRF has enough overlap to fuse meaningfully before truncating.
    candidate_pool = max(top_k * 5, 20)

    try:
        gemini = _get_gemini_client()
        qdrant = get_qdrant_client()
        vector = _embed_texts(gemini, [query], task_type="RETRIEVAL_QUERY")[0]
        hits = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=candidate_pool,
            with_payload=True,
        ).points
    except Exception as exc:
        return {
            "success": False,
            "error": f"{exc}. Have you indexed the dossier yet? Use 'Re-index Transfer Dossier'.",
        }

    dense_ranked: list[tuple[tuple, dict[str, Any]]] = []
    for hit in hits:
        payload = hit.payload or {}
        key = (payload.get("source"), payload.get("page"), payload.get("chunk_index"))
        dense_ranked.append(
            (
                key,
                {
                    "text": payload.get("text", ""),
                    "source": payload.get("source", "unknown"),
                    "page": payload.get("page"),
                    "page_title": payload.get("page_title", ""),
                    "dense_score": round(float(hit.score), 4),
                },
            )
        )

    # Sparse (BM25) side of hybrid search. Safely returns [] if the local
    # cache hasn't been built yet (e.g. before the first "Re-index Transfer
    # Dossier" run) or if `rank_bm25` isn't installed.
    sparse_ranked = _bm25_search(query, candidate_pool)

    if sparse_ranked:
        fused = _reciprocal_rank_fusion(dense_ranked, sparse_ranked)[:top_k]
        results = [
            {
                "text": item["text"],
                "source": item["source"],
                "page": item["page"],
                "page_title": item.get("page_title", ""),
                "score": round(item["rrf_score"], 5),
                "retrieval": "hybrid (dense+BM25 RRF)",
            }
            for item in fused
        ]
    else:
        # Fall back to pure dense search without crashing.
        results = [
            {
                "text": payload["text"],
                "source": payload["source"],
                "page": payload["page"],
                "page_title": payload.get("page_title", ""),
                "score": payload["dense_score"],
                "retrieval": "dense-only (BM25 cache unavailable)",
            }
            for _, payload in dense_ranked[:top_k]
        ]

    return {"success": True, "query": query, "results": results, "count": len(results)}
