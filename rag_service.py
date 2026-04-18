import os
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
    RAG Service using Pinecone (vectors) + Cloudinary (text) + Gemini (LLM).

    Text content is no longer read from MongoDB — it is fetched from the
    Cloudinary URL stored in the Pinecone vector metadata. This keeps MongoDB
    documents lightweight (metadata only) while text lives in Cloudinary.
    """

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

        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        self.pc = Pinecone(api_key=self.pinecone_api_key)

        self.max_context_chars = 80000

        # Similarity threshold — lower = more permissive (catches more docs),
        # higher = stricter (fewer false positives). Tunable via env var.
        self.SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.25"))
        print(f"RAG similarity threshold: {self.SIMILARITY_THRESHOLD}")

        self._greeting_tokens = {
            'hi', 'hello', 'hey', 'howdy', 'hiya', 'yo', 'sup', 'greetings',
            'good morning', 'good afternoon', 'good evening', 'good night',
            'how are you', "how's it going", "what's up", 'whats up',
            "how's everything", 'nice to meet you', 'pleased to meet you',
        }
        self._polite_closers = {
            'thanks', 'thank you', 'thank you so much', 'ok', 'okay',
            'cool', 'great', 'awesome', 'perfect', 'sounds good', 'got it',
            'understood', 'alright', 'sure', 'bye', 'goodbye', 'see you',
            'cheers', 'much appreciated',
        }

    # ── Cloudinary text fetch ────────────────────────────────────────────────

    def _fetch_sections_from_cloudinary(self, url: str) -> list:
        """Fetch the sections JSON uploaded by main.py and return the list."""
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            return resp.json().get("sections", [])
        except Exception as e:
            print(f"Failed to fetch sections from Cloudinary: {e}")
            return []

    def _get_section_text(self, section_number: int, cloudinary_text_url: str) -> str:
        """Fetch all sections and return the text for the specific section_number."""
        sections = self._fetch_sections_from_cloudinary(cloudinary_text_url)
        for s in sections:
            if s.get("section_number") == section_number or sections.index(s) + 1 == section_number:
                return s.get("text_content", "")
        return ""

    # ── Pinecone helpers ─────────────────────────────────────────────────────

    def _get_user_index_name(self, user_id: str) -> str:
        return f"user-{hashlib.md5(user_id.encode()).hexdigest()}"

    def _ensure_user_index(self, user_id: str):
        index_name = self._get_user_index_name(user_id)
        existing = self.pc.list_indexes().names()

        if index_name in existing:
            try:
                desc = self.pc.describe_index(index_name)
                if desc.dimension != self.embedding_dimension:
                    print(f"Index dimension mismatch ({desc.dimension} vs {self.embedding_dimension}). Recreating...")
                    self.pc.delete_index(index_name)
                    existing = []
            except Exception as e:
                print(f"Index check warning: {e}")

        if index_name not in existing:
            self.pc.create_index(
                name=index_name,
                dimension=self.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        return self.pc.Index(index_name)

    # ── Embedding generation ─────────────────────────────────────────────────

    def _generate_embedding(self, text: str) -> np.ndarray:
        if len(text) > 10000:
            parts = [text[:3000], text[len(text)//2-1500:len(text)//2+1500], text[-3000:]]
            text = "\n...\n".join(parts)
        result = genai.embed_content(
            model=self.embedding_model_name,
            content=text,
            task_type="retrieval_document",
        )
        return np.array(result['embedding'], dtype='float32')

    def _generate_query_embedding(self, query: str) -> np.ndarray:
        result = genai.embed_content(
            model=self.embedding_model_name,
            content=query,
            task_type="retrieval_query",
        )
        return np.array(result['embedding'], dtype='float32')

    def _create_indexable_text(self, section: dict) -> str:
        """Build the text string that gets embedded for a section."""
        parts = []
        if section.get("pdf_metadata", {}).get("title"):
            parts.append(f"Document: {section['pdf_metadata']['title']}")
        if section.get("section_title"):
            parts.append(f"Section: {section['section_title']}")
        if section.get("pdf_metadata", {}).get("author"):
            parts.append(f"Author: {section['pdf_metadata']['author']}")
        if section.get("text_content"):
            parts.append(f"\nContent:\n{section['text_content']}")
        return "\n".join(parts)

    def _generate_document_id(self, user_id: str, upload_id: str, section_number: int) -> str:
        return hashlib.md5(f"{user_id}_{upload_id}_{section_number}".encode()).hexdigest()

    def _smart_truncate(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars]
        bp = max(truncated.rfind('.'), truncated.rfind('\n'))
        return truncated[:bp + 1] if bp > max_chars * 0.8 else truncated + "..."

    # ── Conversational intent ────────────────────────────────────────────────

    def _is_conversational(self, query: str) -> bool:
        q = query.strip().lower().rstrip('!.,?')
        if q in self._greeting_tokens or q in self._polite_closers:
            return True
        for phrase in self._greeting_tokens:
            if q.startswith(phrase):
                return True
        words = q.split()
        if len(words) <= 4 and '?' not in query:
            filler = {'i', 'am', 'is', 'are', 'a', 'the', 'just', 'so', 'very'}
            if not [w for w in words if w not in filler]:
                return True
        return False

    def _generate_conversational_response(self, query: str) -> dict:
        try:
            model = genai.GenerativeModel(
                model_name='gemini-2.0-flash',
                system_instruction=(
                    "You are Acumen, a friendly AI assistant on a document Q&A platform. "
                    "Respond warmly and briefly to greetings and casual messages. "
                    "Let users know you're here to help with their uploaded documents."
                )
            )
            response = model.generate_content(
                query,
                generation_config=genai.GenerationConfig(temperature=0.7, max_output_tokens=300)
            )
            return {"success": True, "answer": response.text.strip(), "sources": [],
                    "query": query, "context_documents_used": 0, "response_type": "conversational"}
        except Exception:
            return {"success": True,
                    "answer": "Hello! I'm Acumen. Upload a PDF and ask me anything about it!",
                    "sources": [], "query": query, "response_type": "conversational"}

    def _generate_general_response(self, query: str) -> dict:
        try:
            model = genai.GenerativeModel(
                model_name='gemini-2.0-flash',
                system_instruction=(
                    "You are Acumen, an AI assistant on a document Q&A platform. "
                    "No uploaded documents match this query. Answer from your own knowledge "
                    "and gently remind the user they can upload relevant PDFs for more precise answers."
                )
            )
            response = model.generate_content(
                query,
                generation_config=genai.GenerationConfig(temperature=0.4, max_output_tokens=1024)
            )
            return {"success": True, "answer": response.text.strip(), "sources": [],
                    "query": query, "context_documents_used": 0, "response_type": "general_knowledge"}
        except Exception:
            return {"success": True,
                    "answer": "I don't have uploaded documents on this topic. Upload a relevant PDF for precise answers.",
                    "sources": [], "query": query, "response_type": "general_knowledge"}

    # ── Core RAG methods ─────────────────────────────────────────────────────

    def index_document(
        self,
        user_id: str,
        upload_id: str,
        text_cloudinary_url: Optional[str] = None,
    ) -> dict:
        """
        Index a document into Pinecone.

        Text content is fetched from Cloudinary (not MongoDB).
        The Cloudinary URL is stored in Pinecone metadata so generate_answer
        can fetch text at query time without touching MongoDB.

        If text_cloudinary_url is not provided, it is looked up from MongoDB.
        """
        try:
            # Resolve the Cloudinary URL if not passed directly
            if not text_cloudinary_url:
                sample = self.documents_collection.find_one(
                    {"user_id": user_id, "upload_id": upload_id},
                    {"text_cloudinary_url": 1}
                )
                if not sample:
                    return {"success": False, "message": "No documents found for this upload_id"}
                text_cloudinary_url = sample.get("text_cloudinary_url")

            if not text_cloudinary_url:
                return {"success": False, "message": "No Cloudinary text URL found for this document"}

            # Fetch sections from Cloudinary
            sections = self._fetch_sections_from_cloudinary(text_cloudinary_url)
            if not sections:
                return {"success": False, "message": "Could not fetch sections from Cloudinary"}

            # Fetch MongoDB metadata (section titles, page numbers — lightweight)
            mongo_docs = {
                d["section_number"]: d
                for d in self.documents_collection.find(
                    {"user_id": user_id, "upload_id": upload_id},
                    {"section_number": 1, "section_title": 1, "filename": 1,
                     "pdf_metadata": 1, "start_page": 1, "end_page": 1, "_id": 1}
                )
            }

            print(f"\nIndexing {len(sections)} sections for upload_id: {upload_id}")
            index = self._ensure_user_index(user_id)

            vectors = []
            total_chars = 0

            for idx, section in enumerate(sections):
                section_number = idx + 1
                text = section.get("text_content", "")
                if not text:
                    continue

                # Enrich with MongoDB metadata for this section
                mongo_meta = mongo_docs.get(section_number, {})
                enriched = {
                    **section,
                    "pdf_metadata": mongo_meta.get("pdf_metadata", {}),
                    "section_title": mongo_meta.get("section_title", section.get("section_title", "")),
                }

                indexable = self._create_indexable_text(enriched)
                total_chars += len(indexable)
                embedding = self._generate_embedding(indexable).tolist()
                doc_id = self._generate_document_id(user_id, upload_id, section_number)

                filename = mongo_meta.get("filename", "")
                if not filename and mongo_docs:
                    filename = next(iter(mongo_docs.values())).get("filename", "")

                vectors.append({
                    "id": doc_id,
                    "values": embedding,
                    "metadata": {
                        "upload_id": upload_id,
                        "section_number": section_number,
                        "filename": filename,
                        "section_title": enriched["section_title"],
                        "original_doc_ref": str(mongo_meta.get("_id", "")),
                        "pdf_author": enriched["pdf_metadata"].get("author", ""),
                        "pdf_title": enriched["pdf_metadata"].get("title", ""),
                        "pdf_subject": enriched["pdf_metadata"].get("subject", ""),
                        "char_count": len(text),
                        # Store the Cloudinary URL so generate_answer can fetch
                        # text without going back to MongoDB
                        "text_cloudinary_url": text_cloudinary_url,
                    }
                })
                print(f"  ✓ Section {section_number}: {len(text):,} chars")

            if vectors:
                index.upsert(vectors=vectors, namespace="")
                print(f"✓ Indexed {len(vectors)} sections ({total_chars:,} chars)")

            return {
                "success": True,
                "indexed_sections": len(vectors),
                "total_characters": total_chars,
                "upload_id": upload_id,
                "message": f"Successfully indexed {len(vectors)} sections",
            }

        except Exception as e:
            print(f"Error indexing document: {e}")
            return {"success": False, "message": f"Error indexing document: {str(e)}"}

    def search(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        upload_id: Optional[str] = None,
    ) -> dict:
        try:
            index = self._ensure_user_index(user_id)
            query_embedding = self._generate_query_embedding(query).tolist()

            filter_dict = {"upload_id": upload_id} if upload_id else None

            results = index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                namespace="",
                filter=filter_dict,
            )

            if not results.matches:
                return {"success": False, "message": "No indexed documents available"}

            processed = []
            for match in results.matches:
                meta = match.metadata
                processed.append({
                    "section_number": meta["section_number"],
                    "section_title": meta["section_title"],
                    "filename": meta["filename"],
                    "upload_id": meta["upload_id"],
                    "similarity": float(match.score),
                    "original_doc_ref": meta.get("original_doc_ref", ""),
                    "pdf_author": meta.get("pdf_author", ""),
                    "pdf_title": meta.get("pdf_title", ""),
                    "pdf_subject": meta.get("pdf_subject", ""),
                    "char_count": meta.get("char_count", 0),
                    "text_cloudinary_url": meta.get("text_cloudinary_url", ""),
                })

            return {"success": True, "results": processed, "query": query}

        except Exception as e:
            return {"success": False, "message": f"Search error: {str(e)}"}

    def delete_document_vectors(self, user_id: str, upload_id: str) -> dict:
        try:
            index = self._ensure_user_index(user_id)
            results = index.query(
                vector=[0.0] * self.embedding_dimension,
                top_k=10000,
                filter={"upload_id": upload_id},
                namespace="",
            )
            ids = [m.id for m in results.matches]
            if ids:
                index.delete(ids=ids, namespace="")
            return {"success": True, "deleted_count": len(ids)}
        except Exception as e:
            return {"success": False, "message": f"Delete error: {str(e)}"}

    def delete_user_vectors(self, user_id: str) -> dict:
        try:
            name = self._get_user_index_name(user_id)
            if name in self.pc.list_indexes().names():
                self.pc.delete_index(name)
            return {"success": True, "message": "Deleted all vectors for user"}
        except Exception as e:
            return {"success": False, "message": f"Delete error: {str(e)}"}

    def get_user_stats(self, user_id: str) -> dict:
        try:
            index = self._ensure_user_index(user_id)
            stats = index.describe_index_stats()
            uploads = list(self.documents_collection.aggregate([
                {"$match": {"user_id": user_id}},
                {"$group": {
                    "_id": "$upload_id",
                    "filename": {"$first": "$filename"},
                    "sections_count": {"$sum": 1},
                    "total_chars": {"$sum": "$char_count"},
                }}
            ]))
            return {
                "success": True,
                "total_indexed_sections": stats["total_vector_count"],
                "total_documents": len(uploads),
                "documents": uploads,
            }
        except Exception as e:
            return {"success": False, "message": f"Stats error: {str(e)}"}

    def generate_answer(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        upload_id: Optional[str] = None,
    ) -> dict:
        """
        Generate an answer. Text for context is fetched from Cloudinary URLs
        stored in Pinecone metadata — MongoDB is not consulted for text content.
        """
        try:
            if self._is_conversational(query):
                return self._generate_conversational_response(query)

            search_results = self.search(user_id=user_id, query=query, top_k=top_k, upload_id=upload_id)
            if not search_results["success"]:
                return self._generate_general_response(query)

            results = search_results.get("results", [])
            if not results:
                return self._generate_general_response(query)

            max_similarity = max(r["similarity"] for r in results)
            if max_similarity < self.SIMILARITY_THRESHOLD:
                print(f"Best similarity {max_similarity:.3f} < threshold {self.SIMILARITY_THRESHOLD} — general response")
                return self._generate_general_response(query)

            # ── Build context: fetch text from Cloudinary ──────────────────
            # Group results by their Cloudinary URL to avoid fetching the same
            # JSON file multiple times for documents with several matching sections.
            cloudinary_cache: dict[str, list] = {}
            context_parts = []
            sources = []
            total_context_chars = 0
            max_per_section = self.max_context_chars // max(top_k, 1)

            for i, result in enumerate(results, 1):
                if result["similarity"] < self.SIMILARITY_THRESHOLD:
                    continue

                cl_url = result.get("text_cloudinary_url", "")
                section_number = result["section_number"]

                if cl_url not in cloudinary_cache:
                    cloudinary_cache[cl_url] = self._fetch_sections_from_cloudinary(cl_url)

                sections_list = cloudinary_cache[cl_url]
                # Sections are 0-indexed in the JSON list; section_number is 1-indexed
                text_content = ""
                if sections_list:
                    idx = section_number - 1
                    if 0 <= idx < len(sections_list):
                        text_content = sections_list[idx].get("text_content", "")

                if not text_content:
                    continue

                if len(text_content) > max_per_section:
                    text_content = self._smart_truncate(text_content, max_per_section)

                context_parts.append(
                    f"[Document {i}]\n"
                    f"Source: {result['filename']}\n"
                    f"Section: {result['section_title']}\n"
                    f"Similarity: {result['similarity']:.2f}\n"
                    f"Content:\n{text_content}\n"
                )
                total_context_chars += len(text_content)
                sources.append({
                    "filename": result["filename"],
                    "section_title": result["section_title"],
                    "section_number": section_number,
                    "upload_id": result["upload_id"],
                    "similarity": result["similarity"],
                    "chars_used": len(text_content),
                })

            if not context_parts:
                return self._generate_general_response(query)

            print(f"Context: {total_context_chars:,} chars from {len(sources)} sections")

            system_prompt = """You are Acumen, a highly knowledgeable AI assistant that provides accurate, detailed answers based on PDF documents.

CRITICAL FORMATTING RULES:
- Write in natural, flowing paragraphs
- Use **bold** ONLY to emphasise key terms or names within sentences
- NEVER use bullet points, dashes, or numbered lists unless explicitly asked
- Write conversationally, as if explaining to a colleague

ANSWERING GUIDELINES:
- Use ALL relevant information from the documents
- Reference documents naturally: "According to Document 1..." or "Document 2 explains..."
- Quote important statements using quotation marks when appropriate
- Synthesise information across multiple documents
- Stay factual and cite specific document sections

Remember: Natural prose, detailed explanations, proper citations."""

            user_prompt = (
                f"Answer my question using the document excerpts below.\n\n"
                f"DOCUMENT EXCERPTS:\n{'=' * 60}\n" + "\n".join(context_parts) +
                f"\n\nMY QUESTION: {query}\n\nYour answer:"
            )

            model = genai.GenerativeModel(
                model_name='gemini-2.5-flash',
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


# ── Singleton ────────────────────────────────────────────────────────────────

_rag_service_instance = None


def get_rag_service() -> RAGService:
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = RAGService()
    return _rag_service_instance