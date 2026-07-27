import os
import json
import numpy as np
import requests
from typing import List, Dict, Any, Optional
from pymongo import MongoClient
from dotenv import load_dotenv
import google.generativeai as genai
from datetime import datetime
import hashlib
import ssl
import re
from pinecone import Pinecone, ServerlessSpec
from bson import ObjectId

load_dotenv()


class RAGService:
    """
    RAG Service using Pinecone (vectors) + Cloudinary (chunk text) + Gemini (LLM).

    Documents are stored as overlapping character chunks produced by
    RecursiveCharacterTextSplitter. Each chunk becomes one vector, and the chunk
    text is carried in Pinecone metadata so answers can be built without a
    round-trip to Cloudinary (which is still used as the fallback source).
    """

    SHARED_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "acumen-documents")

    def __init__(self):
        self.mongo_uri = os.getenv("MONGO_URI")
        self.database_name = os.getenv("DATABASE_NAME")
        self.documents_collection_name = os.getenv("DOCUMENTS_COLLECTION", "pdf_documents")

        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        self.client = MongoClient(self.mongo_uri, tls=True, tlsAllowInvalidCertificates=True)
        self.db = self.client[self.database_name]
        self.documents_collection = self.db[self.documents_collection_name]

        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        genai.configure(api_key=self.gemini_api_key)

        self.embedding_model_name = "models/gemini-embedding-001"
        self.embedding_dimension = 3072
        self.embed_batch_size = int(os.getenv("EMBED_BATCH_SIZE", "32"))
        self.upsert_batch_size = int(os.getenv("UPSERT_BATCH_SIZE", "50"))

        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        self.pc = Pinecone(api_key=self.pinecone_api_key)

        self.max_context_chars = 80000
        self.max_metadata_chars = 8000
        self.max_history_turns = int(os.getenv("MAX_HISTORY_TURNS", "6"))

        self.SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.25"))
        print(f"RAG similarity threshold: {self.SIMILARITY_THRESHOLD}")

    # ── Cloudinary ───────────────────────────────────────────────────────────

    def _fetch_chunks_from_cloudinary(self, url: str) -> list:
        """Returns a normalised chunk list. Supports the legacy 'sections' payload."""
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"Failed to fetch chunks from Cloudinary: {e}")
            return []

        if data.get("chunks"):
            return data["chunks"]

        legacy = data.get("sections", [])
        return [
            {
                "chunk_index": i + 1,
                "text": s.get("text_content", ""),
                "chunk_title": s.get("section_title", f"Section {i + 1}"),
                "start_page": s.get("start_page", 1),
                "end_page": s.get("end_page", 1),
                "char_count": s.get("char_count", len(s.get("text_content", ""))),
            }
            for i, s in enumerate(legacy)
        ]

    # ── Pinecone helpers ─────────────────────────────────────────────────────

    def _get_user_namespace(self, user_id: str) -> str:
        return f"user-{hashlib.md5(user_id.encode()).hexdigest()}"

    def _vector_id(self, upload_id: str, chunk_index: int) -> str:
        return f"{upload_id}#{chunk_index}"

    def _ensure_user_index(self, user_id: str):
        existing = self.pc.list_indexes().names()

        if self.SHARED_INDEX_NAME not in existing:
            print(f"Creating shared Pinecone index '{self.SHARED_INDEX_NAME}' ...")
            self.pc.create_index(
                name=self.SHARED_INDEX_NAME,
                dimension=self.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        else:
            try:
                desc = self.pc.describe_index(self.SHARED_INDEX_NAME)
                if desc.dimension != self.embedding_dimension:
                    raise ValueError(
                        f"Existing index dimension {desc.dimension} != expected "
                        f"{self.embedding_dimension}. Delete '{self.SHARED_INDEX_NAME}' "
                        "manually or set a new PINECONE_INDEX_NAME env var."
                    )
            except ValueError:
                raise
            except Exception as e:
                print(f"Index check warning: {e}")

        return self.pc.Index(self.SHARED_INDEX_NAME)

    # ── Embeddings ───────────────────────────────────────────────────────────

    def _embed_texts(self, texts: List[str], task_type: str = "retrieval_document") -> List[list]:
        embeddings: List[list] = []
        for i in range(0, len(texts), self.embed_batch_size):
            batch = texts[i:i + self.embed_batch_size]
            try:
                result = genai.embed_content(
                    model=self.embedding_model_name,
                    content=batch,
                    task_type=task_type,
                )
                embeddings.extend(result["embedding"])
            except Exception as e:
                print(f"Batch embedding failed ({e}) — falling back to single requests")
                for text in batch:
                    single = genai.embed_content(
                        model=self.embedding_model_name,
                        content=text,
                        task_type=task_type,
                    )
                    embeddings.append(single["embedding"])
        return embeddings

    def _generate_query_embedding(self, query: str) -> np.ndarray:
        result = genai.embed_content(
            model=self.embedding_model_name,
            content=query,
            task_type="retrieval_query",
        )
        return np.array(result['embedding'], dtype='float32')

    def _create_indexable_text(self, chunk: dict, doc_meta: dict) -> str:
        parts = []
        if doc_meta.get("filename"):
            parts.append(f"Document: {doc_meta['filename']}")
        if doc_meta.get("pdf_metadata", {}).get("title"):
            parts.append(f"Title: {doc_meta['pdf_metadata']['title']}")
        if doc_meta.get("pdf_metadata", {}).get("author"):
            parts.append(f"Author: {doc_meta['pdf_metadata']['author']}")
        if chunk.get("chunk_title"):
            parts.append(f"Location: {chunk['chunk_title']}")
        parts.append(f"\n{chunk.get('text', '')}")
        return "\n".join(parts)

    def _smart_truncate(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars]
        bp = max(truncated.rfind('.'), truncated.rfind('\n'))
        return truncated[:bp + 1] if bp > max_chars * 0.8 else truncated + "..."

    # ── Conversation context ─────────────────────────────────────────────────

    def _format_history(self, history: Optional[List[dict]]) -> str:
        """Trim and flatten recent turns. Assistant turns are cut harder — they
        are long, and only their gist is needed to resolve a reference."""
        if not history:
            return ""

        lines = []
        for turn in history[-self.max_history_turns:]:
            role = "User" if str(turn.get("role", "")).strip().lower() == "user" else "Acumen"
            content = " ".join(str(turn.get("content", "")).split())
            if not content:
                continue
            limit = 1200 if role == "User" else 800
            if len(content) > limit:
                content = content[:limit].rstrip() + "..."
            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    # ── Fallback when retrieval finds nothing relevant ───────────────────────

    def _generate_general_response(self, query: str, history: Optional[List[dict]] = None) -> dict:
        try:
            model = genai.GenerativeModel(
                model_name='gemini-3.5-flash-lite',
                system_instruction=(
                    "You are Acumen, an AI assistant on a document Q&A platform. "
                    "No uploaded documents match this query. Answer from your own knowledge "
                    "and gently remind the user they can upload relevant PDFs for more precise answers."
                )
            )
            conversation = self._format_history(history)
            prompt = f"Conversation so far:\n{conversation}\n\nCurrent message: {query}" if conversation else query
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(temperature=0.4, max_output_tokens=1024)
            )
            return {"success": True, "answer": response.text.strip(), "sources": [],
                    "query": query, "context_documents_used": 0, "response_type": "general_knowledge"}
        except Exception:
            return {"success": True,
                    "answer": "I don't have uploaded documents on this topic. Upload a relevant PDF for precise answers.",
                    "sources": [], "query": query, "response_type": "general_knowledge"}

    # ── Indexing ─────────────────────────────────────────────────────────────

    def index_document(
        self,
        user_id: str,
        upload_id: str,
        text_cloudinary_url: Optional[str] = None,
    ) -> dict:
        try:
            doc_meta = self.documents_collection.find_one(
                {"user_id": user_id, "upload_id": upload_id},
                {"filename": 1, "pdf_metadata": 1, "text_cloudinary_url": 1, "_id": 1}
            ) or {}

            if not text_cloudinary_url:
                text_cloudinary_url = doc_meta.get("text_cloudinary_url")
            if not text_cloudinary_url:
                return {"success": False, "message": "No Cloudinary text URL found for this document"}

            chunks = self._fetch_chunks_from_cloudinary(text_cloudinary_url)
            chunks = [c for c in chunks if c.get("text", "").strip()]
            if not chunks:
                return {"success": False, "message": "Could not fetch chunks from Cloudinary"}

            index = self._ensure_user_index(user_id)
            namespace = self._get_user_namespace(user_id)
            self.delete_document_vectors(user_id, upload_id)

            print(f"Indexing {len(chunks)} chunks for upload_id: {upload_id}")

            indexable = [self._create_indexable_text(c, doc_meta) for c in chunks]
            embeddings = self._embed_texts(indexable)

            filename = doc_meta.get("filename", "")
            pdf_meta = doc_meta.get("pdf_metadata", {}) or {}
            vectors = []

            for chunk, embedding in zip(chunks, embeddings):
                chunk_index = chunk.get("chunk_index", len(vectors) + 1)
                vectors.append({
                    "id": self._vector_id(upload_id, chunk_index),
                    "values": embedding,
                    "metadata": {
                        "upload_id": upload_id,
                        "chunk_index": chunk_index,
                        "section_number": chunk_index,
                        "section_title": chunk.get("chunk_title", f"Chunk {chunk_index}"),
                        "filename": filename,
                        "original_doc_ref": str(doc_meta.get("_id", "")),
                        "pdf_author": pdf_meta.get("author", ""),
                        "pdf_title": pdf_meta.get("title", ""),
                        "pdf_subject": pdf_meta.get("subject", ""),
                        "start_page": chunk.get("start_page", 1),
                        "end_page": chunk.get("end_page", 1),
                        "char_count": chunk.get("char_count", len(chunk["text"])),
                        "text": chunk["text"] if len(chunk["text"]) <= self.max_metadata_chars else "",
                        "text_cloudinary_url": text_cloudinary_url,
                    }
                })

            for i in range(0, len(vectors), self.upsert_batch_size):
                index.upsert(vectors=vectors[i:i + self.upsert_batch_size], namespace=namespace)

            total_chars = sum(v["metadata"]["char_count"] for v in vectors)
            print(f"Indexed {len(vectors)} chunks ({total_chars:,} chars)")

            return {
                "success": True,
                "indexed_chunks": len(vectors),
                "indexed_sections": len(vectors),
                "total_characters": total_chars,
                "upload_id": upload_id,
                "message": f"Successfully indexed {len(vectors)} chunks",
            }

        except Exception as e:
            print(f"Error indexing document: {e}")
            return {"success": False, "message": f"Error indexing document: {str(e)}"}

    # ── Retrieval ────────────────────────────────────────────────────────────

    def _route_query(self, query: str, history: Optional[List[dict]] = None) -> dict:
        """
        One model call decides whether a message needs the user's documents and,
        when it does not, writes the reply itself. When it does, it returns the
        retrieval variants — so routing costs nothing beyond the query expansion
        that already ran on every question.

        Recent turns are supplied so that references ("it", "the second one",
        "ok") are resolved against what was actually said, and so that the
        retrieval variants are self-contained.
        """
        fallback = {"needs_documents": True, "queries": [query], "reply": ""}
        try:
            model = genai.GenerativeModel(
                model_name='gemini-3.5-flash-lite',
                system_instruction=(
                    "You route incoming messages for Acumen, an assistant that answers "
                    "questions about a user's uploaded PDF documents.\n\n"
                    "Reply with JSON containing exactly these keys: needs_documents "
                    "(boolean), queries (array of strings), reply (string).\n\n"
                    "You may be given the recent conversation. Use it only to work out what "
                    "the current message refers to — never to answer from it.\n\n"
                    "Set needs_documents to false ONLY when the current message asks for no "
                    "information at all — greetings, thanks, acknowledgements that close a "
                    "topic, farewells, or small talk about you. In that case leave queries "
                    "empty and write reply yourself, speaking as Acumen: warm, one or two "
                    "sentences, in the user's own language and register. Never invent "
                    "document contents in a reply.\n\n"
                    "Set needs_documents to true for everything else, including general "
                    "knowledge questions, bare topics, single words, and any short message "
                    "that continues a line of questioning rather than closing it — an "
                    "acknowledgement followed by a request is a request. When in doubt, "
                    "choose true.\n\n"
                    "When needs_documents is true, leave reply empty and fill queries with 3 "
                    "short keyword-rich search phrases approaching the intent from different "
                    "angles: broad, specific, and synonym-based. Each phrase must stand on "
                    "its own with every pronoun and reference replaced by what it points to "
                    "in the conversation, because they are matched against documents that "
                    "have no knowledge of this chat. Never assume a document type. Do not "
                    "write full sentences in queries."
                )
            )

            conversation = self._format_history(history)
            prompt = (
                f"Recent conversation:\n{conversation}\n\nCurrent message: {query}"
                if conversation else query
            )

            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=400,
                    response_mime_type="application/json",
                )
            )

            raw = response.text.strip()
            if raw.startswith("```"):
                raw = raw.strip("`")
                raw = raw[4:] if raw.lower().startswith("json") else raw
            data = json.loads(raw.strip())

            needs_documents = bool(data.get("needs_documents", True))
            queries = [str(q).strip() for q in data.get("queries", []) if str(q).strip()]
            reply = str(data.get("reply", "")).strip()

            if not needs_documents and reply:
                print(f"Routed as conversational: {query!r}")
                return {"needs_documents": False, "queries": [], "reply": reply}

            if not queries:
                queries = [query.strip()]
            elif not conversation and query.strip() not in queries:
                queries.insert(0, query.strip())

            print(f"Routed for retrieval: {queries[:4]}")
            return {"needs_documents": True, "queries": queries[:4], "reply": ""}

        except Exception as e:
            print(f"Routing failed, falling back to retrieval: {e}")
            return fallback

    def search(
        self,
        user_id: str,
        query: str,
        top_k: int = 12,
        upload_id: Optional[str] = None,
        queries: Optional[List[str]] = None,
    ) -> dict:
        try:
            index = self._ensure_user_index(user_id)
            namespace = self._get_user_namespace(user_id)
            filter_dict = {"upload_id": upload_id} if upload_id else None

            if not queries:
                queries = self._route_query(query).get("queries") or [query]
            best_by_id: dict = {}

            for q in queries:
                embedding = self._generate_query_embedding(q).tolist()
                res = index.query(
                    vector=embedding,
                    top_k=top_k,
                    include_metadata=True,
                    namespace=namespace,
                    filter=filter_dict,
                )
                for match in res.matches:
                    existing = best_by_id.get(match.id)
                    if existing is None or match.score > existing.score:
                        best_by_id[match.id] = match

            merged = sorted(best_by_id.values(), key=lambda m: m.score, reverse=True)
            if not merged:
                return {"success": False, "message": "No indexed documents available"}

            candidate_uploads = {
                m.metadata.get("upload_id") for m in merged if m.metadata.get("upload_id")
            }
            live_uploads = self._active_upload_ids(user_id, candidate_uploads)
            stale = candidate_uploads - live_uploads
            if stale:
                print(f"Ignoring {len(stale)} deleted document(s) still present in the index: {stale}")
                merged = [m for m in merged if m.metadata.get("upload_id") in live_uploads]
                if not merged:
                    return {"success": False, "message": "No indexed documents available"}

            merged = merged[:top_k]

            processed = []
            for match in merged:
                meta = match.metadata
                processed.append({
                    "chunk_index": int(meta.get("chunk_index", meta.get("section_number", 0))),
                    "section_number": int(meta.get("section_number", 0)),
                    "section_title": meta.get("section_title", ""),
                    "filename": meta.get("filename", ""),
                    "upload_id": meta.get("upload_id", ""),
                    "similarity": float(match.score),
                    "start_page": int(meta.get("start_page", 0)),
                    "end_page": int(meta.get("end_page", 0)),
                    "pdf_author": meta.get("pdf_author", ""),
                    "pdf_title": meta.get("pdf_title", ""),
                    "pdf_subject": meta.get("pdf_subject", ""),
                    "char_count": int(meta.get("char_count", 0)),
                    "text": meta.get("text", ""),
                    "text_cloudinary_url": meta.get("text_cloudinary_url", ""),
                })

            return {"success": True, "results": processed, "query": query}

        except Exception as e:
            return {"success": False, "message": f"Search error: {str(e)}"}

    # ── Deletion & stats ─────────────────────────────────────────────────────

    @staticmethod
    def _as_id_list(page) -> List[str]:
        if isinstance(page, str):
            return [page]
        if isinstance(page, list):
            return [v if isinstance(v, str) else getattr(v, "id", str(v)) for v in page]
        vectors = getattr(page, "vectors", None)
        if vectors:
            return [getattr(v, "id", str(v)) for v in vectors]
        return [str(page)]

    def _scan_namespace_for_upload(self, index, namespace: str, upload_id: str) -> List[str]:
        """Walk every vector in the namespace and match on metadata. Only used when
        the ID-prefix lookup finds nothing, i.e. vectors written by an older build
        whose IDs did not encode the upload_id."""
        found: List[str] = []
        try:
            for page in index.list(namespace=namespace):
                batch = self._as_id_list(page)
                if not batch:
                    continue
                fetched = index.fetch(ids=batch, namespace=namespace)
                vectors = getattr(fetched, "vectors", None)
                if vectors is None and isinstance(fetched, dict):
                    vectors = fetched.get("vectors", {})
                for vid, vec in (vectors or {}).items():
                    meta = getattr(vec, "metadata", None)
                    if meta is None and isinstance(vec, dict):
                        meta = vec.get("metadata")
                    if (meta or {}).get("upload_id") == upload_id:
                        found.append(vid)
        except Exception as e:
            print(f"Namespace scan failed for {upload_id}: {e}")
        return found

    def delete_document_vectors(self, user_id: str, upload_id: str) -> dict:
        """Deletes by ID because Pinecone serverless does not support delete-by-filter.
        Reports how many vectors were removed and whether any survived, so callers can
        refuse to drop the Mongo record while vectors are still searchable."""
        try:
            index = self._ensure_user_index(user_id)
            namespace = self._get_user_namespace(user_id)

            ids: List[str] = []
            try:
                for page in index.list(prefix=f"{upload_id}#", namespace=namespace):
                    ids.extend(self._as_id_list(page))
            except Exception as e:
                print(f"Prefix listing unavailable ({e}); falling back to scan")

            if not ids:
                ids = self._scan_namespace_for_upload(index, namespace, upload_id)

            ids = list(dict.fromkeys(ids))
            if not ids:
                return {"success": True, "deleted_count": 0, "remaining": 0,
                        "message": "No vectors found for this document"}

            for i in range(0, len(ids), 1000):
                index.delete(ids=ids[i:i + 1000], namespace=namespace)

            remaining = 0
            try:
                check = index.fetch(ids=ids[:100], namespace=namespace)
                vectors = getattr(check, "vectors", None)
                if vectors is None and isinstance(check, dict):
                    vectors = check.get("vectors", {})
                remaining = len(vectors or {})
            except Exception as e:
                print(f"Post-delete verification skipped: {e}")

            return {
                "success": remaining == 0,
                "deleted_count": len(ids),
                "remaining": remaining,
                "message": (
                    f"Deleted {len(ids)} vectors"
                    if remaining == 0
                    else f"{remaining} vectors still present after delete (eventual consistency or partial failure)"
                ),
            }
        except Exception as e:
            return {"success": False, "deleted_count": 0, "message": f"Delete error: {str(e)}"}

    def _active_upload_ids(self, user_id: str, upload_ids: set) -> set:
        if not upload_ids:
            return set()
        rows = self.documents_collection.find(
            {"user_id": user_id, "upload_id": {"$in": list(upload_ids)}},
            {"upload_id": 1, "_id": 0},
        )
        return {r["upload_id"] for r in rows}

    def cleanup_orphan_vectors(self, user_id: str) -> dict:
        """Removes vectors whose document no longer exists in Mongo."""
        try:
            index = self._ensure_user_index(user_id)
            namespace = self._get_user_namespace(user_id)

            live = {
                d["_id"] for d in self.documents_collection.aggregate([
                    {"$match": {"user_id": user_id}},
                    {"$group": {"_id": "$upload_id"}},
                ])
            }

            orphan_ids: List[str] = []
            orphan_uploads: set = set()
            for page in index.list(namespace=namespace):
                batch = self._as_id_list(page)
                if not batch:
                    continue
                fetched = index.fetch(ids=batch, namespace=namespace)
                vectors = getattr(fetched, "vectors", None)
                if vectors is None and isinstance(fetched, dict):
                    vectors = fetched.get("vectors", {})
                for vid, vec in (vectors or {}).items():
                    meta = getattr(vec, "metadata", None)
                    if meta is None and isinstance(vec, dict):
                        meta = vec.get("metadata")
                    uid = (meta or {}).get("upload_id")
                    if uid and uid not in live:
                        orphan_ids.append(vid)
                        orphan_uploads.add(uid)

            for i in range(0, len(orphan_ids), 1000):
                index.delete(ids=orphan_ids[i:i + 1000], namespace=namespace)

            return {
                "success": True,
                "deleted_vectors": len(orphan_ids),
                "orphaned_documents": len(orphan_uploads),
                "message": f"Removed {len(orphan_ids)} orphaned vectors from {len(orphan_uploads)} deleted documents",
            }
        except Exception as e:
            return {"success": False, "message": f"Cleanup error: {str(e)}"}

    def delete_user_vectors(self, user_id: str) -> dict:
        try:
            index = self._ensure_user_index(user_id)
            namespace = self._get_user_namespace(user_id)
            index.delete(delete_all=True, namespace=namespace)
            return {"success": True, "message": "Deleted all vectors for user"}
        except Exception as e:
            return {"success": False, "message": f"Delete error: {str(e)}"}

    def get_user_stats(self, user_id: str) -> dict:
        try:
            index = self._ensure_user_index(user_id)
            namespace = self._get_user_namespace(user_id)
            stats = index.describe_index_stats()
            ns_stats = stats.get("namespaces", {}).get(namespace, {})
            total_vectors = ns_stats.get("vector_count", 0)
            uploads = list(self.documents_collection.aggregate([
                {"$match": {"user_id": user_id}},
                {"$group": {
                    "_id": "$upload_id",
                    "filename": {"$first": "$filename"},
                    "chunks_count": {"$first": "$total_chunks"},
                    "total_chars": {"$sum": {"$ifNull": ["$total_characters", "$char_count"]}},
                }}
            ]))
            return {
                "success": True,
                "total_indexed_chunks": total_vectors,
                "total_indexed_sections": total_vectors,
                "total_documents": len(uploads),
                "documents": uploads,
            }
        except Exception as e:
            return {"success": False, "message": f"Stats error: {str(e)}"}

    # ── Answer generation ────────────────────────────────────────────────────

    def _resolve_chunk_text(self, result: dict, cache: dict) -> str:
        if result.get("text"):
            return result["text"]

        url = result.get("text_cloudinary_url", "")
        if not url:
            return ""
        if url not in cache:
            cache[url] = self._fetch_chunks_from_cloudinary(url)

        chunks = cache[url]
        target = result.get("chunk_index") or result.get("section_number")
        for chunk in chunks:
            if chunk.get("chunk_index") == target:
                return chunk.get("text", "")

        idx = (target or 1) - 1
        if 0 <= idx < len(chunks):
            return chunks[idx].get("text", "")
        return ""

    def generate_answer(
        self,
        user_id: str,
        query: str,
        top_k: int = 12,
        upload_id: Optional[str] = None,
        history: Optional[List[dict]] = None,
    ) -> dict:
        try:
            route = self._route_query(query, history)
            if not route["needs_documents"]:
                return {
                    "success": True,
                    "answer": route["reply"],
                    "sources": [],
                    "query": query,
                    "context_documents_used": 0,
                    "response_type": "conversational",
                }

            search_results = self.search(
                user_id=user_id, query=query, top_k=top_k,
                upload_id=upload_id, queries=route["queries"],
            )
            if not search_results["success"]:
                return self._generate_general_response(query, history)

            results = search_results.get("results", [])
            if not results:
                return self._generate_general_response(query, history)

            max_similarity = max(r["similarity"] for r in results)
            if max_similarity < self.SIMILARITY_THRESHOLD:
                print(f"Best similarity {max_similarity:.3f} < threshold {self.SIMILARITY_THRESHOLD}")
                return self._generate_general_response(query, history)

            cloudinary_cache: dict = {}
            context_parts = []
            sources = []
            total_context_chars = 0

            for result in results:
                if result["similarity"] < self.SIMILARITY_THRESHOLD:
                    continue

                text_content = self._resolve_chunk_text(result, cloudinary_cache)
                if not text_content:
                    continue

                if total_context_chars + len(text_content) > self.max_context_chars:
                    text_content = self._smart_truncate(
                        text_content, max(0, self.max_context_chars - total_context_chars)
                    )
                    if not text_content:
                        break

                location = result.get("section_title") or ""
                context_parts.append(
                    f"Source: {result['filename']}\n"
                    f"Location: {location}\n"
                    f"Content:\n{text_content}\n"
                )
                total_context_chars += len(text_content)
                sources.append({
                    "filename": result["filename"],
                    "section_title": location,
                    "section_number": result.get("section_number", 0),
                    "chunk_index": result.get("chunk_index", 0),
                    "start_page": result.get("start_page", 0),
                    "end_page": result.get("end_page", 0),
                    "upload_id": result["upload_id"],
                    "similarity": result["similarity"],
                    "chars_used": len(text_content),
                })

            if not context_parts:
                return self._generate_general_response(query, history)

            print(f"Context: {total_context_chars:,} chars from {len(sources)} chunks")

            system_prompt = """You are Acumen, a highly knowledgeable AI assistant that answers questions based on the user's uploaded PDF documents.

CRITICAL FORMATTING RULES:
- Write in natural, flowing paragraphs
- Use **bold** ONLY to emphasise key terms or names within sentences
- NEVER use bullet points, dashes, or numbered lists unless explicitly asked
- Write conversationally, like a knowledgeable friend explaining something — not like a report

ANSWERING GUIDELINES:
- The excerpts are overlapping passages from the same documents, so ignore repetition and merge them into one coherent answer
- Do NOT reference excerpt numbers, chunk indexes or page markers unless the user asks where something came from
- If you need to attribute something, mention the filename or document title naturally
- Quote important phrases using quotation marks when it adds value
- Stay factual and grounded in the provided content
- Give complete, detailed answers — do not truncate or summarise unnecessarily
- When earlier turns are provided, treat the question as a continuation: resolve what it refers to, keep your wording consistent with what you already said, and do not repeat an explanation the user has just been given

Remember: Sound like a person who has read the documents, not a system citing its sources."""

            conversation = self._format_history(history)
            user_prompt = (
                (f"CONVERSATION SO FAR:\n{conversation}\n\n" if conversation else "") +
                f"Answer my question using the document excerpts below.\n\n"
                f"DOCUMENT EXCERPTS:\n{'=' * 60}\n" + "\n".join(context_parts) +
                f"\n\nMY QUESTION: {query}\n\nYour answer:"
            )

            model = genai.GenerativeModel(
                model_name='gemini-3.5-flash-lite',
                system_instruction=system_prompt,
            )
            response = model.generate_content(
                user_prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.1, top_p=0.1, top_k=5,
                    candidate_count=1, max_output_tokens=4096,
                )
            )

            answer = response.text
            answer = re.sub(r'^\s*[\*\-]\s+', '', answer, flags=re.MULTILINE)
            answer = re.sub(r'^\s*\d+\.\s+', '', answer, flags=re.MULTILINE)
            answer = re.sub(r'\n\s*\n\s*\n+', '\n\n', answer).strip()

            return {
                "success": True,
                "answer": answer,
                "sources": sources,
                "query": query,
                "context_documents_used": len(sources),
                "total_context_chars": total_context_chars,
                "response_type": "rag",
            }

        except Exception as e:
            print(f"Error generating answer: {e}")
            return {"success": False, "message": f"Error generating answer: {str(e)}", "answer": None}


_rag_service_instance = None


def get_rag_service() -> RAGService:
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = RAGService()
    return _rag_service_instance