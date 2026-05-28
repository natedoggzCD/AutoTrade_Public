"""
YouTube RAG Manager - vector store management for YouTube intelligence.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

try:
    import chromadb
    _CHROMA_ERROR = ""
except Exception as e:  # pragma: no cover - optional dependency guard
    chromadb = None
    _CHROMA_ERROR = str(e)

try:
    from sentence_transformers import SentenceTransformer
    _ST_ERROR = ""
except Exception as e:  # pragma: no cover - optional dependency guard
    SentenceTransformer = None
    _ST_ERROR = str(e)

# Optional llama-index backend (best effort only).
_LLAMA_AVAILABLE = False
_LLAMA_ERROR = ""
Document = None
VectorStoreIndex = None
StorageContext = None
ServiceContext = None
ChromaVectorStore = None
HuggingFaceEmbedding = None

try:
    from llama_index.core import Document, VectorStoreIndex, StorageContext
    try:
        from llama_index.core import ServiceContext
    except Exception:
        ServiceContext = None
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    _LLAMA_AVAILABLE = True
except Exception as e:
    _LLAMA_ERROR = str(e)

logger = logging.getLogger("AutoTrade.YouTubeRAG")

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAG_DATA_DIR = PROJECT_ROOT / "data" / "youtube" / "rag"
VECTOR_STORE_DIR = RAG_DATA_DIR / "vectorstore"
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)


class YouTubeRAGManager:
    """
    Manages semantic search across YouTube transcripts and intelligence.

    Backend selection:
    1) llama-index + Chroma (if import/runtime is healthy)
    2) native Chroma + SentenceTransformer fallback
    """

    def __init__(self, collection_name: str = "youtube_intelligence"):
        self.collection_name = collection_name
        self._available = False
        self._disabled_reason = ""
        self._backend = "disabled"
        self.chroma_client = None
        self.chroma_collection = None
        self.vector_store = None
        self.storage_context = None
        self.embed_model = None
        self.service_context = None

        if chromadb is None:
            self._disabled_reason = f"chromadb_missing: {_CHROMA_ERROR}"
            logger.warning(
                "YouTubeRAGManager disabled (%s)",
                self._disabled_reason,
            )
            return

        try:
            self.chroma_client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                collection_name
            )
        except Exception as e:
            self._disabled_reason = f"chroma_init_failed: {e}"
            logger.warning(
                "YouTubeRAGManager disabled (%s)",
                self._disabled_reason,
            )
            return

        if _LLAMA_AVAILABLE:
            try:
                self._init_llama_backend()
                self._available = True
                self._backend = "llama_index"
                logger.info(
                    "YouTubeRAGManager initialized (backend=%s, collection=%s)",
                    self._backend,
                    collection_name,
                )
                return
            except Exception as e:
                logger.warning(
                    "Llama-index backend unavailable, falling back to native Chroma (%s)",
                    e,
                )

        if SentenceTransformer is None:
            self._disabled_reason = (
                "native_embedder_missing"
                + (f": {_ST_ERROR}" if _ST_ERROR else "")
                + (f" | llama_error: {_LLAMA_ERROR}" if _LLAMA_ERROR else "")
            )
            logger.warning(
                "YouTubeRAGManager disabled (%s)",
                self._disabled_reason,
            )
            return

        try:
            self.embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self._available = True
            self._backend = "native_chroma"
            logger.info(
                "YouTubeRAGManager initialized (backend=%s, collection=%s)",
                self._backend,
                collection_name,
            )
        except Exception as e:
            self._disabled_reason = f"native_embedder_init_failed: {e}"
            logger.warning(
                "YouTubeRAGManager disabled (%s)",
                self._disabled_reason,
            )

    def _init_llama_backend(self) -> None:
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.service_context = None
        if ServiceContext is not None:
            try:
                self.service_context = ServiceContext.from_defaults(
                    embed_model=self.embed_model,
                    llm=None,
                )
            except Exception:
                self.service_context = None

    def _llama_index_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"storage_context": self.storage_context}
        if self.service_context is not None:
            kwargs["service_context"] = self.service_context
        else:
            kwargs["embed_model"] = self.embed_model
        return kwargs

    def _llama_query_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if self.service_context is not None:
            kwargs["service_context"] = self.service_context
        else:
            kwargs["embed_model"] = self.embed_model
        return kwargs

    def _native_upsert(self, text: str, metadata: Dict[str, Any]) -> bool:
        try:
            embedding = self.embed_model.encode(
                [text],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )[0].tolist()
            doc_id = (
                f"{metadata.get('type', 'entry')}:"
                f"{metadata.get('date', datetime.now().strftime('%Y-%m-%d'))}:"
                f"{uuid4().hex[:12]}"
            )
            self.chroma_collection.upsert(
                ids=[doc_id],
                documents=[text],
                metadatas=[metadata],
                embeddings=[embedding],
            )
            return True
        except Exception as e:
            logger.error(f"Native upsert failed: {e}")
            return False

    def index_transcript(
        self,
        ticker: str,
        date_str: str,
        transcript: str,
        metadata: Dict[str, Any],
    ) -> bool:
        """Chunk and index a full transcript."""
        if not self._available:
            logger.debug("Skipping transcript indexing (RAG disabled)")
            return False
        if self._backend == "llama_index":
            try:
                doc = Document(
                    text=transcript,
                    metadata={
                        **metadata,
                        "ticker": ticker,
                        "date": date_str,
                        "type": "transcript",
                    },
                )
                VectorStoreIndex.from_documents([doc], **self._llama_index_kwargs())
                logger.info(f"Indexed transcript for {ticker} on {date_str}")
                return True
            except Exception as e:
                logger.error(f"Failed to index transcript (llama backend): {e}")
                return False

        ok = self._native_upsert(
            text=transcript,
            metadata={
                **metadata,
                "ticker": ticker,
                "date": date_str,
                "type": "transcript",
            },
        )
        if ok:
            logger.info(f"Indexed transcript for {ticker} on {date_str}")
        return ok

    def index_extraction(self, ticker: str, date_str: str, extraction: Dict[str, Any]) -> bool:
        """Index flattened extraction JSON."""
        if not self._available:
            logger.debug("Skipping extraction indexing (RAG disabled)")
            return False

        text = f"Extraction for {ticker} on {date_str}:\n" + json.dumps(extraction, indent=2)
        metadata = {
            "ticker": ticker,
            "date": date_str,
            "type": "extraction",
            "sentiment": extraction.get("sentiment", "neutral"),
        }

        if self._backend == "llama_index":
            try:
                doc = Document(text=text, metadata=metadata)
                VectorStoreIndex.from_documents([doc], **self._llama_index_kwargs())
                logger.info(f"Indexed extraction for {ticker} on {date_str}")
                return True
            except Exception as e:
                logger.error(f"Failed to index extraction (llama backend): {e}")
                return False

        ok = self._native_upsert(text=text, metadata=metadata)
        if ok:
            logger.info(f"Indexed extraction for {ticker} on {date_str}")
        return ok

    def query(self, query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant insights."""
        if not self._available:
            logger.debug("Skipping query (RAG disabled)")
            return []

        if self._backend == "llama_index":
            try:
                index = VectorStoreIndex.from_vector_store(
                    self.vector_store,
                    **self._llama_query_kwargs(),
                )
                query_engine = index.as_query_engine(similarity_top_k=limit)
                response = query_engine.query(query_text)
                return [
                    {
                        "text": node.node.get_content(),
                        "metadata": node.node.metadata,
                        "score": node.score,
                    }
                    for node in response.source_nodes
                ]
            except Exception as e:
                logger.error(f"Query failed (llama backend): {e}")
                return []

        try:
            q_embedding = self.embed_model.encode(
                [query_text],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )[0].tolist()
            raw = self.chroma_collection.query(
                query_embeddings=[q_embedding],
                n_results=max(1, int(limit)),
                include=["documents", "metadatas", "distances"],
            )
            docs = (raw.get("documents") or [[]])[0]
            metas = (raw.get("metadatas") or [[]])[0]
            dists = (raw.get("distances") or [[]])[0]
            results: List[Dict[str, Any]] = []
            for doc, meta, dist in zip(docs, metas, dists):
                try:
                    score = max(0.0, 1.0 - float(dist))
                except Exception:
                    score = 0.0
                results.append(
                    {
                        "text": doc,
                        "metadata": meta or {},
                        "score": score,
                    }
                )
            return results
        except Exception as e:
            logger.error(f"Query failed (native backend): {e}")
            return []


_rag_manager = None


def get_rag_manager() -> YouTubeRAGManager:
    global _rag_manager
    if _rag_manager is None:
        _rag_manager = YouTubeRAGManager()
    return _rag_manager
