import os
import numpy as np
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import google.generativeai as genai
from datetime import datetime
import hashlib
import re
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

class RAGService:
    """
    RAG (Retrieval-Augmented Generation) Service using Pinecone and Gemini.
    - Embedding model: gemini-embedding-001 (dim=3072)
    - Conversational queries are answered directly without RAG
    - Document queries use full RAG pipeline
    - Text content is stored directly in Pinecone vector metadata (no MongoDB)
    """

    def __init__(self):
        # Configure Gemini
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        genai.configure(api_key=self.gemini_api_key)

        self.embedding_model_name = "models/gemini-embedding-001"
        self.embedding_dimension = 3072

        # Pinecone configuration
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")

        self.pc = Pinecone(api_key=self.pinecone_api_key)

        # Context management
        self.max_context_chars = 80000

        # Chunking configuration
        self.index_chunk_size    = 1500   # characters per Pinecone vector
        self.index_chunk_overlap = 200    # overlap to preserve boundary context
        self.min_chunk_size      = 150    # sub-chunks smaller than this are skipped

        # Conversational intent keywords
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

    # ── helpers ─────────────────────────────────────────────────────────────

    def _split_into_chunks(self, text: str) -> List[str]:
        """
        Split text into overlapping character-based chunks that respect
        natural language boundaries (paragraph → sentence → line → word).
        """
        text = text.strip()
        if not text or len(text) < self.min_chunk_size:
            return []

        if len(text) <= self.index_chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0

        while start < len(text):
            end = min(start + self.index_chunk_size, len(text))

            if end == len(text):
                chunk = text[start:].strip()
                if len(chunk) >= self.min_chunk_size:
                    chunks.append(chunk)
                break

            break_point = -1
            for sep in ['\n\n', '. ', '\n', ' ']:
                bp = text.rfind(sep, start, end)
                if bp != -1 and bp > start:
                    break_point = bp + len(sep)
                    break

            if break_point == -1:
                break_point = end

            chunk = text[start:break_point].strip()
            if len(chunk) >= self.min_chunk_size:
                chunks.append(chunk)

            start = max(start + 1, break_point - self.index_chunk_overlap)

            snap_end = min(start + self.index_chunk_overlap // 2, len(text))
            for sep in ['\n\n', '\n', '. ']:
                bp = text.find(sep, start, snap_end)
                if bp != -1:
                    start = bp + len(sep)
                    break

        return chunks

    def _get_user_index_name(self, user_id: str) -> str:
        user_hash = hashlib.md5(user_id.encode()).hexdigest()
        return f"user-{user_hash}"

    def _ensure_user_index(self, user_id: str):
        """Create or retrieve the user's Pinecone index (3072-dim)."""
        index_name = self._get_user_index_name(user_id)
        existing = self.pc.list_indexes().names()

        if index_name in existing:
            try:
                desc = self.pc.describe_index(index_name)
                if desc.dimension != self.embedding_dimension:
                    print(
                        f"Index '{index_name}' has dimension {desc.dimension} "
                        f"but model requires {self.embedding_dimension}. "
                        "Deleting and recreating …"
                    )
                    self.pc.delete_index(index_name)
                    existing = []
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
        """Generate an embedding for a single text chunk (gemini-embedding-001)."""
        try:
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

    def _create_searchable_text(self, section: Dict[str, Any]) -> str:
        """Build the enriched text used for embedding (title + metadata header + content)."""
        parts = []
        meta = section.get('pdf_metadata', {})
        if meta.get('title'):
            parts.append(f"Document: {meta['title']}")
        if section.get('section_title'):
            parts.append(f"Section: {section['section_title']}")
        if meta.get('author'):
            parts.append(f"Author: {meta['author']}")
        if meta.get('subject'):
            parts.append(f"Subject: {meta['subject']}")
        if section.get('text_content'):
            parts.append(f"\nContent:\n{section['text_content']}")
        return "\n".join(parts)

    def _generate_section_id(self, user_id: str, upload_id: str, section_number: int) -> str:
        """Generate a stable, unique ID for a section (used as deduplication key)."""
        unique_string = f"{user_id}_{upload_id}_{section_number}"
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

    # ── Conversational intent detection ─────────────────────────────────────

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
            content_words = [w for w in words if w not in filler]
            if not content_words:
                return True

        return False

    def _generate_conversational_response(self, query: str) -> Dict[str, Any]:
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

    # ── Core RAG methods ─────────────────────────────────────────────────────

    def index_document(
        self,
        user_id: str,
        upload_id: str,
        sections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Index document sections directly into Pinecone.

        CHANGED: sections are passed in directly (no MongoDB read).
        Each section dict must have: section_number, section_title, filename,
        text_content, and optionally pdf_metadata (dict with author/title/subject).

        Text content is stored in Pinecone vector metadata so it can be
        retrieved at query time without any secondary database lookup.
        """
        try:
            if not sections:
                return {"success": False, "message": "No sections provided to index"}

            print(f"\nIndexing {len(sections)} sections for upload_id: {upload_id}")

            index = self._ensure_user_index(user_id)

            vectors = []
            total_chars = 0
            total_chunks = 0

            for section in sections:
                if not section.get('text_content'):
                    print(f"  Skipping section {section.get('section_number')} - no content")
                    continue

                # Build enriched text for embedding (includes title + metadata header)
                searchable_text = self._create_searchable_text(section)

                # Split into overlapping sub-chunks
                sub_chunks = self._split_into_chunks(searchable_text)
                if not sub_chunks:
                    print(f"  Skipping section {section.get('section_number')} - below min chunk size")
                    continue

                section_id = self._generate_section_id(
                    user_id, upload_id, section['section_number']
                )
                section_chars = len(section.get('text_content', ''))
                total_chars += section_chars
                meta = section.get('pdf_metadata', {})

                for chunk_idx, chunk_text in enumerate(sub_chunks):
                    embedding = self._generate_embedding(chunk_text).tolist()

                    vector_id = f"{section_id}_c{chunk_idx}"

                    vectors.append({
                        "id": vector_id,
                        "values": embedding,
                        "metadata": {
                            "upload_id": upload_id,
                            "section_number": section['section_number'],
                            "filename": section['filename'],
                            "section_title": section['section_title'],
                            "section_id": section_id,          # for deduplication
                            "text_content": chunk_text,         # stored here — no MongoDB needed
                            "pdf_author": meta.get('author', ''),
                            "pdf_title": meta.get('title', ''),
                            "pdf_subject": meta.get('subject', ''),
                            "char_count": section_chars,
                            "chunk_index": chunk_idx,
                            "total_chunks": len(sub_chunks),
                        }
                    })
                    total_chunks += 1

                print(
                    f"  ✓ Section {section['section_number']}: "
                    f"{section_chars} chars → {len(sub_chunks)} sub-chunk(s)"
                )

            if vectors:
                batch_size = 100
                for i in range(0, len(vectors), batch_size):
                    index.upsert(vectors=vectors[i:i + batch_size], namespace="")

                print(
                    f"\n✓ Indexed {len(sections)} sections as {total_chunks} "
                    f"sub-chunks ({total_chars:,} total characters)"
                )

            return {
                "success": True,
                "indexed_sections": len(sections),
                "indexed_chunks": total_chunks,
                "total_characters": total_chars,
                "upload_id": upload_id,
                "message": (
                    f"Successfully indexed {len(sections)} sections "
                    f"as {total_chunks} sub-chunks in Pinecone"
                )
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
        """
        Semantic search through a user's Pinecone index.

        CHANGED: deduplication now uses section_id (instead of original_doc_ref).
        Results include text_content from Pinecone metadata directly.
        """
        try:
            index = self._ensure_user_index(user_id)

            query_embedding = self._generate_query_embedding(query).tolist()

            filter_dict = {}
            if upload_id:
                filter_dict["upload_id"] = upload_id

            raw_top_k = top_k * 4

            results = index.query(
                vector=query_embedding,
                top_k=raw_top_k,
                include_metadata=True,
                namespace="",
                filter=filter_dict if filter_dict else None
            )

            if not results.matches:
                return {"success": False, "message": "No indexed documents available"}

            # Deduplicate: keep best-scoring chunk per section
            seen: Dict[str, Any] = {}   # section_id → best match so far
            for match in results.matches:
                sid = match.metadata.get('section_id', match.id)
                if sid not in seen or match.score > seen[sid].score:
                    seen[sid] = match

            deduped = sorted(seen.values(), key=lambda m: m.score, reverse=True)[:top_k]

            processed_results = []
            for match in deduped:
                meta = match.metadata
                processed_results.append({
                    "section_number": meta['section_number'],
                    "section_title": meta['section_title'],
                    "filename": meta['filename'],
                    "upload_id": meta['upload_id'],
                    "similarity": float(match.score),
                    "section_id": meta.get('section_id', ''),
                    "text_content": meta.get('text_content', ''),   # available directly from Pinecone
                    "pdf_author": meta.get('pdf_author', ''),
                    "pdf_title": meta.get('pdf_title', ''),
                    "pdf_subject": meta.get('pdf_subject', ''),
                    "char_count": meta.get('char_count', 0),
                    "chunk_index": meta.get('chunk_index', 0),
                    "total_chunks": meta.get('total_chunks', 1),
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
        """Return Pinecone index stats for the user."""
        try:
            index = self._ensure_user_index(user_id)
            stats = index.describe_index_stats()
            total_vectors = stats['total_vector_count']

            return {
                "success": True,
                "total_indexed_vectors": total_vectors,
            }

        except Exception as e:
            return {"success": False, "message": f"Error getting stats: {str(e)}"}

    # ── Answer generation ────────────────────────────────────────────────────

    def generate_answer(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        upload_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate an answer for any user query.

        CHANGED: text_content is read directly from Pinecone search results.
        No secondary MongoDB lookup is performed.

        Decision flow:
          1. Conversational / greeting  →  direct Gemini response (no RAG)
          2. No indexed documents        →  general Gemini response + nudge to upload
          3. Low similarity scores       →  general Gemini response (query is off-topic)
          4. Good document match         →  full RAG response with citations
        """
        try:
            # Step 1: Handle conversational queries immediately
            if self._is_conversational(query):
                return self._generate_conversational_response(query)

            # Step 2: Semantic search in Pinecone
            search_results = self.search(
                user_id=user_id,
                query=query,
                top_k=top_k,
                upload_id=upload_id
            )

            if not search_results["success"]:
                return self._generate_general_response(query)

            results = search_results.get("results", [])

            if not results:
                return self._generate_general_response(query)

            # Step 3: Check similarity threshold
            SIMILARITY_THRESHOLD = 0.40
            max_similarity = max(r["similarity"] for r in results)
            if max_similarity < SIMILARITY_THRESHOLD:
                return self._generate_general_response(query)

            # Step 4: Full RAG pipeline — text comes from Pinecone metadata directly
            context_parts = []
            sources = []
            total_context_chars = 0
            max_per_section = self.max_context_chars // top_k

            for i, result in enumerate(results, 1):
                # CHANGED: text_content is already in the search result — no MongoDB needed
                text_content = result.get('text_content', '')

                if not text_content:
                    continue

                if len(text_content) > max_per_section:
                    text_content = self._smart_truncate_context(text_content, max_per_section)

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