import os
import numpy as np
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

# Load environment variables
load_dotenv()

class RAGService:
    """
    RAG (Retrieval-Augmented Generation) Service using Pinecone and Gemini.
    - Embedding model: gemini-embedding-001 (dim=3072)
    - Conversational queries are answered directly without RAG
    - Document queries use full RAG pipeline
    """

    def __init__(self):
        # MongoDB configuration
        self.mongo_uri = os.getenv("MONGO_URI")
        self.database_name = os.getenv("DATABASE_NAME")
        self.documents_collection_name = os.getenv("DOCUMENTS_COLLECTION", "pdf_documents")

        # Initialize MongoDB
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.minimum_version = ssl.TLSVersion.TLSv1_2

        self.client = MongoClient(
            self.mongo_uri,
            tls=True,
            tlsAllowInvalidCertificates=True
        )

        self.db = self.client[self.database_name]
        self.documents_collection = self.db[self.documents_collection_name]

        # Configure Gemini
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        genai.configure(api_key=self.gemini_api_key)

        # ── FIX 1: Updated embedding model ──────────────────────────────────
        # gemini-embedding-001 produces 3072-dimensional vectors.
        # NOTE: Any existing Pinecone indexes built with text-embedding-004
        # (768-dim) are incompatible. They will be recreated automatically
        # the first time _ensure_user_index() is called for each user.
        # Users must re-index their documents after this change.
        self.embedding_model_name = "models/gemini-embedding-001"
        self.embedding_dimension = 3072
        # ────────────────────────────────────────────────────────────────────

        # Pinecone configuration
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")

        self.pc = Pinecone(api_key=self.pinecone_api_key)

        # Context management
        self.max_context_chars = 80000

        # ── FIX 2: Conversational intent keywords ────────────────────────────
        self._greeting_tokens = {
            'hi', 'hello', 'hey', 'howdy', 'hiya', 'yo', 'sup', 'greetings',
            'good morning', 'good afternoon', 'good evening', 'good night',
            'how are you', "how's it going", 'what\'s up', 'whats up',
            "how's everything", 'nice to meet you', 'pleased to meet you',
        }
        self._polite_closers = {
            'thanks', 'thank you', 'thank you so much', 'ok', 'okay',
            'cool', 'great', 'awesome', 'perfect', 'sounds good', 'got it',
            'understood', 'alright', 'sure', 'bye', 'goodbye', 'see you',
            'cheers', 'much appreciated',
        }
        # ────────────────────────────────────────────────────────────────────

    # ── helpers ─────────────────────────────────────────────────────────────

    def _get_user_index_name(self, user_id: str) -> str:
        user_hash = hashlib.md5(user_id.encode()).hexdigest()
        return f"user-{user_hash}"

    def _ensure_user_index(self, user_id: str):
        """Create or retrieve the user's Pinecone index (3072-dim)."""
        index_name = self._get_user_index_name(user_id)

        existing = self.pc.list_indexes().names()

        if index_name in existing:
            # Check whether the existing index has the right dimension.
            # If it was created with the old model (768) we delete and recreate.
            try:
                idx = self.pc.Index(index_name)
                stats = idx.describe_index_stats()
                # Pinecone doesn't expose dimension in stats directly,
                # but we can probe via the index description.
                desc = self.pc.describe_index(index_name)
                if desc.dimension != self.embedding_dimension:
                    print(
                        f"Index '{index_name}' has dimension {desc.dimension} "
                        f"but model requires {self.embedding_dimension}. "
                        "Deleting and recreating …"
                    )
                    self.pc.delete_index(index_name)
                    existing = []  # force recreation below
            except Exception as e:
                print(f"Warning during index dimension check: {e}")

        if index_name not in existing:
            self.pc.create_index(
                name=index_name,
                dimension=self.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

        return self.pc.Index(index_name)

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for document indexing (gemini-embedding-001)."""
        try:
            if len(text) > 10000:
                parts = [
                    text[:3000],
                    text[len(text) // 2 - 1500: len(text) // 2 + 1500],
                    text[-3000:]
                ]
                text = "\n...\n".join(parts)

            result = genai.embed_content(
                model=self.embedding_model_name,
                content=text,
                task_type="retrieval_document"
            )
            return np.array(result['embedding'], dtype='float32')
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            raise

    def _generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a search query (gemini-embedding-001)."""
        try:
            result = genai.embed_content(
                model=self.embedding_model_name,
                content=query,
                task_type="retrieval_query"
            )
            return np.array(result['embedding'], dtype='float32')
        except Exception as e:
            print(f"Error generating query embedding: {str(e)}")
            raise

    def _create_document_text(self, document: Dict[str, Any]) -> str:
        parts = []
        if document.get('pdf_title'):
            parts.append(f"Document: {document['pdf_title']}")
        if document.get('section_title'):
            parts.append(f"Section: {document['section_title']}")
        if document.get('pdf_author'):
            parts.append(f"Author: {document['pdf_author']}")
        if document.get('pdf_subject'):
            parts.append(f"Subject: {document['pdf_subject']}")
        if document.get('text_content'):
            parts.append(f"\nContent:\n{document['text_content']}")
        return "\n".join(parts)

    def _generate_document_id(self, document: Dict[str, Any]) -> str:
        unique_string = (
            f"{document['user_id']}_{document['upload_id']}_"
            f"{document['section_number']}"
        )
        return hashlib.md5(unique_string.encode()).hexdigest()

    def _smart_truncate_context(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars]
        last_period = truncated.rfind('.')
        last_newline = truncated.rfind('\n')
        break_point = max(last_period, last_newline)
        if break_point > max_chars * 0.8:
            return truncated[:break_point + 1]
        return truncated + "..."

    # ── FIX 2: Conversational-intent detection ───────────────────────────────

    def _is_conversational(self, query: str) -> bool:
        """
        Return True when the query is a greeting, social pleasantry, or
        a short polite phrase that doesn't need document retrieval.
        """
        q = query.strip().lower().rstrip('!.,?')

        # Exact match against known phrases
        if q in self._greeting_tokens or q in self._polite_closers:
            return True

        # Starts-with match for multi-word greetings
        for phrase in self._greeting_tokens:
            if q.startswith(phrase):
                return True

        # Very short messages (≤ 4 words) with no question mark are likely conversational
        words = q.split()
        if len(words) <= 4 and '?' not in query:
            filler = {'i', 'am', 'is', 'are', 'a', 'the', 'just', 'so', 'very'}
            content_words = [w for w in words if w not in filler]
            if not content_words:
                return True

        return False

    def _generate_conversational_response(self, query: str) -> Dict[str, Any]:
        """
        Produce a friendly, direct response for conversational queries
        using Gemini without any document context.
        """
        try:
            system_prompt = (
                "You are Acumen, a friendly and professional AI assistant "
                "embedded in a document Q&A platform. When users greet you or "
                "make casual conversation, respond warmly and briefly. "
                "Let them know you're here to help them explore their uploaded "
                "documents — but always keep the tone natural and human. "
                "Never be stiff or robotic."
            )

            model = genai.GenerativeModel(
                model_name='gemini-2.0-flash',
                system_instruction=system_prompt
            )
            response = model.generate_content(
                query,
                generation_config=genai.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=300,
                )
            )

            return {
                "success": True,
                "answer": response.text.strip(),
                "sources": [],
                "query": query,
                "context_documents_used": 0,
                "total_context_chars": 0,
                "response_type": "conversational"
            }
        except Exception as e:
            return {
                "success": True,
                "answer": (
                    "Hello! I'm Acumen, your document assistant. "
                    "Feel free to upload a PDF and ask me anything about it!"
                ),
                "sources": [],
                "query": query,
                "response_type": "conversational"
            }

    def _generate_general_response(self, query: str) -> Dict[str, Any]:
        """
        Answer a non-conversational query that has no relevant document matches.
        Gemini answers from its own knowledge and gracefully notes the context.
        """
        try:
            system_prompt = (
                "You are Acumen, an intelligent AI assistant on a document Q&A "
                "platform. The user has asked a question, but either no documents "
                "have been uploaded yet or none of the uploaded documents contain "
                "relevant information. "
                "Answer the question from your own knowledge as helpfully as "
                "possible. If the topic is very niche or you're uncertain, say so "
                "honestly. At the end, gently remind the user that you can give "
                "much more precise, source-backed answers if they upload relevant "
                "documents. Keep the tone warm and professional."
            )

            model = genai.GenerativeModel(
                model_name='gemini-2.0-flash',
                system_instruction=system_prompt
            )
            response = model.generate_content(
                query,
                generation_config=genai.GenerationConfig(
                    temperature=0.4,
                    max_output_tokens=1024,
                )
            )

            return {
                "success": True,
                "answer": response.text.strip(),
                "sources": [],
                "query": query,
                "context_documents_used": 0,
                "total_context_chars": 0,
                "response_type": "general_knowledge"
            }
        except Exception as e:
            return {
                "success": True,
                "answer": (
                    "That's a great question! I don't have any uploaded documents "
                    "on this topic right now, so I can't give you a source-backed "
                    "answer. Try uploading a relevant PDF and I'll be able to help "
                    "you much more precisely."
                ),
                "sources": [],
                "query": query,
                "response_type": "general_knowledge"
            }

    # ── core RAG methods ─────────────────────────────────────────────────────

    def index_document(self, user_id: str, upload_id: str) -> Dict[str, Any]:
        """Index document sections into Pinecone."""
        try:
            documents = list(self.documents_collection.find({
                "user_id": user_id,
                "upload_id": upload_id
            }).sort("section_number", 1))

            if not documents:
                return {"success": False, "message": "No documents found for this upload_id"}

            print(f"\nIndexing {len(documents)} sections for upload_id: {upload_id}")

            index = self._ensure_user_index(user_id)

            vectors = []
            total_chars = 0

            for doc in documents:
                if not doc.get('text_content'):
                    print(f"  Skipping section {doc.get('section_number')} - no content")
                    continue

                searchable_text = self._create_document_text(doc)
                total_chars += len(searchable_text)

                embedding = self._generate_embedding(searchable_text).tolist()
                doc_id = self._generate_document_id(doc)

                vectors.append({
                    "id": doc_id,
                    "values": embedding,
                    "metadata": {
                        "upload_id": upload_id,
                        "section_number": doc['section_number'],
                        "filename": doc['filename'],
                        "section_title": doc['section_title'],
                        "original_doc_ref": str(doc['_id']),
                        "pdf_author": doc.get('pdf_metadata', {}).get('author', ''),
                        "pdf_title": doc.get('pdf_metadata', {}).get('title', ''),
                        "pdf_subject": doc.get('pdf_metadata', {}).get('subject', ''),
                        "char_count": len(doc.get('text_content', ''))
                    }
                })

                print(f"  ✓ Section {doc['section_number']}: {len(doc.get('text_content', ''))} chars")

            if vectors:
                index.upsert(vectors=vectors, namespace="")
                print(f"\n✓ Indexed {len(vectors)} sections ({total_chars:,} total characters)")

            return {
                "success": True,
                "indexed_sections": len(vectors),
                "total_characters": total_chars,
                "upload_id": upload_id,
                "message": f"Successfully indexed {len(vectors)} sections in Pinecone"
            }

        except Exception as e:
            print(f"Error indexing document: {str(e)}")
            return {"success": False, "message": f"Error indexing document: {str(e)}"}

    def search(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        upload_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Semantic search through a user's Pinecone index."""
        try:
            index = self._ensure_user_index(user_id)

            query_embedding = self._generate_query_embedding(query).tolist()

            filter_dict = {}
            if upload_id:
                filter_dict["upload_id"] = upload_id

            results = index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                namespace="",
                filter=filter_dict if filter_dict else None
            )

            if not results.matches:
                return {"success": False, "message": "No indexed documents available"}

            processed_results = []
            for match in results.matches:
                meta = match.metadata
                processed_results.append({
                    "section_number": meta['section_number'],
                    "section_title": meta['section_title'],
                    "filename": meta['filename'],
                    "upload_id": meta['upload_id'],
                    "similarity": float(match.score),
                    "original_doc_ref": meta['original_doc_ref'],
                    "pdf_author": meta.get('pdf_author', ''),
                    "pdf_title": meta.get('pdf_title', ''),
                    "pdf_subject": meta.get('pdf_subject', ''),
                    "char_count": meta.get('char_count', 0)
                })

            return {"success": True, "results": processed_results, "query": query}

        except Exception as e:
            return {"success": False, "message": f"Error during search: {str(e)}"}

    def delete_document_vectors(self, user_id: str, upload_id: str) -> Dict[str, Any]:
        try:
            index = self._ensure_user_index(user_id)

            results = index.query(
                vector=[0] * self.embedding_dimension,
                top_k=10000,
                filter={"upload_id": upload_id},
                namespace=""
            )

            ids_to_delete = [match.id for match in results.matches]
            if ids_to_delete:
                index.delete(ids=ids_to_delete, namespace="")

            return {"success": True, "deleted_count": len(ids_to_delete)}

        except Exception as e:
            return {"success": False, "message": f"Error deleting document vectors: {str(e)}"}

    def delete_user_vectors(self, user_id: str) -> Dict[str, Any]:
        try:
            index_name = self._get_user_index_name(user_id)
            if index_name in self.pc.list_indexes().names():
                self.pc.delete_index(index_name)
            return {"success": True, "message": "Deleted all vectors for user"}
        except Exception as e:
            return {"success": False, "message": f"Error deleting user vectors: {str(e)}"}

    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        try:
            index = self._ensure_user_index(user_id)
            stats = index.describe_index_stats()
            total_sections = stats['total_vector_count']

            pipeline = [
                {"$match": {"user_id": user_id}},
                {"$group": {
                    "_id": "$upload_id",
                    "filename": {"$first": "$filename"},
                    "sections_count": {"$sum": 1},
                    "total_chars": {"$sum": "$char_count"}
                }}
            ]

            uploads = list(self.documents_collection.aggregate(pipeline))

            return {
                "success": True,
                "total_indexed_sections": total_sections,
                "total_documents": len(uploads),
                "documents": uploads
            }

        except Exception as e:
            return {"success": False, "message": f"Error getting stats: {str(e)}"}

    # ── FIX 2: Upgraded generate_answer with conversational handling ──────────

    def generate_answer(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        upload_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate an answer for any user query.

        Decision flow:
          1. Conversational / greeting  →  direct Gemini response (no RAG)
          2. No indexed documents        →  general Gemini response + nudge to upload
          3. Low similarity scores       →  general Gemini response (query is off-topic)
          4. Good document match         →  full RAG response with citations
        """
        try:
            # ── Step 1: Handle conversational queries immediately ──────────
            if self._is_conversational(query):
                return self._generate_conversational_response(query)

            # ── Step 2: Try RAG search ────────────────────────────────────
            search_results = self.search(
                user_id=user_id,
                query=query,
                top_k=top_k,
                upload_id=upload_id
            )

            # No documents indexed at all → answer from general knowledge
            if not search_results["success"]:
                return self._generate_general_response(query)

            results = search_results.get("results", [])

            # No matches returned
            if not results:
                return self._generate_general_response(query)

            # ── Step 3: Check similarity threshold ────────────────────────
            # If the best match is below 0.40 the query is probably unrelated
            # to the uploaded documents — fall back to general knowledge.
            SIMILARITY_THRESHOLD = 0.40
            max_similarity = max(r["similarity"] for r in results)
            if max_similarity < SIMILARITY_THRESHOLD:
                return self._generate_general_response(query)

            # ── Step 4: Full RAG pipeline ─────────────────────────────────
            context_parts = []
            sources = []
            total_context_chars = 0
            max_per_section = self.max_context_chars // top_k

            for i, result in enumerate(results, 1):
                try:
                    doc = self.documents_collection.find_one(
                        {"_id": ObjectId(result["original_doc_ref"])}
                    )
                    text_content = doc.get('text_content', "") if doc else ""
                except Exception:
                    text_content = ""

                if not text_content:
                    continue

                if len(text_content) > max_per_section:
                    text_content = self._smart_truncate_context(
                        text_content, max_per_section
                    )

                context_parts.append(
                    f"[Document {i}]\n"
                    f"Source: {result['filename']}\n"
                    f"Section: {result['section_title']}\n"
                    f"Similarity: {result['similarity']:.2f}\n"
                    f"Content:\n{text_content}\n"
                )

                total_context_chars += len(text_content)

                sources.append({
                    "filename": result['filename'],
                    "section_title": result['section_title'],
                    "section_number": result['section_number'],
                    "upload_id": result['upload_id'],
                    "similarity": result['similarity'],
                    "chars_used": len(text_content)
                })

            if not context_parts:
                return self._generate_general_response(query)

            context = "\n" + "=" * 60 + "\n".join(context_parts)

            print(f"\nContext built: {total_context_chars:,} characters from {len(sources)} sections")

            system_prompt = """You are Acumen, a highly knowledgeable AI assistant that provides accurate, detailed answers based on PDF documents.

CRITICAL FORMATTING RULES:
- Write in natural, flowing paragraphs
- Use **bold** (double asterisks) ONLY to emphasise key terms, concepts, or names within sentences
- NEVER use bullet points, dashes, or numbered lists unless explicitly asked
- Write conversationally, as if explaining to a colleague

ANSWERING GUIDELINES:
- Provide comprehensive, detailed answers using ALL relevant information from the documents
- Reference documents naturally: "According to Document 1..." or "Document 2 explains..."
- Quote important statements when appropriate, using quotation marks
- If multiple documents discuss the same topic, synthesise the information
- If information is missing or unclear, acknowledge this honestly
- Stay factual and cite specific document sections

QUALITY STANDARDS:
- Depth: Provide thorough explanations, not surface-level summaries
- Accuracy: Use only information from the provided documents
- Clarity: Organise information logically, but in paragraph form
- Context: Help the reader understand WHY something matters, not just WHAT it says

Remember: Natural prose, detailed explanations, proper citations."""

            user_prompt = f"""Answer my question using the document excerpts below.

DOCUMENT EXCERPTS:
{context}

MY QUESTION: {query}

Please provide a comprehensive, well-explained answer that:
1. Uses ALL relevant information from the documents
2. Cites which documents support your statements
3. Explains concepts thoroughly
4. Maintains a natural, conversational tone
5. Uses paragraphs, not lists (unless I specifically asked for a list)

Your answer:"""

            model = genai.GenerativeModel(
                model_name='gemini-2.0-flash',
                system_instruction=system_prompt
            )

            generation_config = genai.GenerationConfig(
                temperature=0.1,
                top_p=0.1,
                top_k=5,
                candidate_count=1,
                max_output_tokens=4096,
            )

            response = model.generate_content(
                user_prompt,
                generation_config=generation_config
            )

            answer = response.text

            # Clean up any stray list formatting artifacts
            answer = re.sub(r'^\s*[\*\-]\s+', '', answer, flags=re.MULTILINE)
            answer = re.sub(r'^\s*\d+\.\s+', '', answer, flags=re.MULTILINE)
            answer = re.sub(r'\n\s*\n\s*\n+', '\n\n', answer)
            answer = answer.strip()

            return {
                "success": True,
                "answer": answer,
                "sources": sources,
                "query": query,
                "context_documents_used": len(sources),
                "total_context_chars": total_context_chars,
                "response_type": "rag"
            }

        except Exception as e:
            print(f"Error generating answer: {str(e)}")
            return {
                "success": False,
                "message": f"Error generating answer: {str(e)}",
                "answer": None
            }


# Singleton instance
_rag_service_instance = None


def get_rag_service() -> RAGService:
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = RAGService()
    return _rag_service_instance