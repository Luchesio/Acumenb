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
    IMPROVED RAG (Retrieval-Augmented Generation) Service using Pinecone and Gemini
    - Full content extraction (no truncation)
    - Better context window management
    - Improved answer generation
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
        
        # Initialize embedding model
        self.embedding_model_name = "models/text-embedding-004"
        
        # Embedding dimension
        self.embedding_dimension = 768
        
        # Pinecone configuration
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        
        # Context management
        self.max_context_chars = 80000  # Increased for full content
    
    def _get_user_index_name(self, user_id: str) -> str:
        """Generate a unique, valid index name for each user"""
        user_hash = hashlib.md5(user_id.encode()).hexdigest()
        return f"user-{user_hash}"
    
    def _ensure_user_index(self, user_id: str):
        """Create or get the user's Pinecone index"""
        index_name = self._get_user_index_name(user_id)
        
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=self.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        
        return self.pc.Index(index_name)
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for document indexing"""
        try:
            # Truncate if too long (Gemini has limits)
            if len(text) > 10000:
                # For very long sections, take beginning, middle, and end
                parts = [
                    text[:3000],
                    text[len(text)//2 - 1500:len(text)//2 + 1500],
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
        """Generate embedding for search query"""
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
        """Create searchable text from document with proper structure"""
        parts = []
        
        # Add metadata for context
        if document.get('pdf_title'):
            parts.append(f"Document: {document['pdf_title']}")
        
        if document.get('section_title'):
            parts.append(f"Section: {document['section_title']}")
        
        if document.get('pdf_author'):
            parts.append(f"Author: {document['pdf_author']}")
        
        if document.get('pdf_subject'):
            parts.append(f"Subject: {document['pdf_subject']}")
        
        # Add full content (no truncation)
        if document.get('text_content'):
            parts.append(f"\nContent:\n{document['text_content']}")
        
        return "\n".join(parts)
    
    def _generate_document_id(self, document: Dict[str, Any]) -> str:
        """Generate unique ID for document section"""
        unique_string = f"{document['user_id']}_{document['upload_id']}_{document['section_number']}"
        return hashlib.md5(unique_string.encode()).hexdigest()
    
    def index_document(self, user_id: str, upload_id: str) -> Dict[str, Any]:
        """Index document with FULL content (no truncation)"""
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
                
                # Create searchable text with FULL content
                searchable_text = self._create_document_text(doc)
                total_chars += len(searchable_text)
                
                # Generate embedding
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
    
    def search(self, user_id: str, query: str, top_k: int = 5, upload_id: Optional[str] = None) -> Dict[str, Any]:
        """Search with improved result handling"""
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
        """Delete vectors for a specific document"""
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
        """Delete all vectors for a user"""
        try:
            index_name = self._get_user_index_name(user_id)
            if index_name in self.pc.list_indexes().names():
                self.pc.delete_index(index_name)
            return {"success": True, "message": "Deleted all vectors for user"}
            
        except Exception as e:
            return {"success": False, "message": f"Error deleting user vectors: {str(e)}"}
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics about indexed documents"""
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
    
    def _smart_truncate_context(self, text: str, max_chars: int) -> str:
        """Intelligently truncate text while preserving meaning"""
        if len(text) <= max_chars:
            return text
        
        # Try to break at sentence boundaries
        truncated = text[:max_chars]
        last_period = truncated.rfind('.')
        last_newline = truncated.rfind('\n')
        
        break_point = max(last_period, last_newline)
        
        if break_point > max_chars * 0.8:  # If we can keep at least 80%
            return truncated[:break_point + 1]
        
        return truncated + "..."
    
    def generate_answer(self, user_id: str, query: str, top_k: int = 5, upload_id: Optional[str] = None) -> Dict[str, Any]:
        """
        IMPROVED: Generate answer using FULL content from retrieved sections
        """
        try:
            # Search for relevant sections
            search_results = self.search(user_id=user_id, query=query, top_k=top_k, upload_id=upload_id)
            
            if not search_results["success"]:
                return {"success": False, "message": search_results.get("message", "Search failed"), "answer": None}
            
            if not search_results["results"]:
                return {
                    "success": True,
                    "answer": "I couldn't find any relevant information in your documents to answer this question. Please try rephrasing your query or upload relevant documents.",
                    "sources": [],
                    "query": query
                }
            
            # Build context from retrieved sections with FULL content
            context_parts = []
            sources = []
            total_context_chars = 0
            max_per_section = self.max_context_chars // top_k  # Distribute evenly
            
            for i, result in enumerate(search_results["results"], 1):
                # Fetch FULL text_content from MongoDB
                try:
                    doc = self.documents_collection.find_one({"_id": ObjectId(result["original_doc_ref"])})
                    text_content = doc.get('text_content', "") if doc else ""
                except:
                    text_content = ""
                
                if not text_content:
                    continue
                
                # Smart truncation if needed (but try to use as much as possible)
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
            
            context = "\n" + "="*60 + "\n".join(context_parts)
            
            print(f"\nContext built: {total_context_chars:,} characters from {len(sources)} sections")
            
            # Improved system prompt
            system_prompt = """You are a highly knowledgeable AI assistant that provides accurate, detailed answers based on PDF documents.

CRITICAL FORMATTING RULES:
- Write in natural, flowing paragraphs
- Use **bold** (double asterisks) ONLY to emphasize key terms, concepts, or names within sentences
- NEVER use bullet points, dashes, or numbered lists unless explicitly asked
- Write conversationally, as if explaining to a colleague

ANSWERING GUIDELINES:
- Provide comprehensive, detailed answers using ALL relevant information from the documents
- Reference documents naturally: "According to Document 1..." or "Document 2 explains..."
- Quote important statements when appropriate, using quotation marks
- If multiple documents discuss the same topic, synthesize the information
- If information is missing or unclear, acknowledge this honestly
- Stay factual and cite specific document sections

QUALITY STANDARDS:
- Depth: Provide thorough explanations, not surface-level summaries
- Accuracy: Use only information from the provided documents
- Clarity: Organize information logically, but in paragraph form
- Context: Help the reader understand WHY something matters, not just WHAT it says

Remember: Natural prose, detailed explanations, proper citations."""

            user_prompt = f"""I need you to answer my question using the document excerpts below. These excerpts contain the full relevant content from my documents.

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

            # Use Gemini to generate answer
            model = genai.GenerativeModel(
                model_name='gemini-3-flash-preview',
                system_instruction=system_prompt
            )

            generation_config = genai.GenerationConfig(
                temperature=0.1,  # Lower for more focused answers
                top_p=0.1,
                top_k=5,
                candidate_count=1,
                max_output_tokens=4096,  # Allow longer responses
            )
            
            response = model.generate_content(
                user_prompt,
                generation_config=generation_config
            )
            
            answer = response.text
            
            # Clean up any remaining list formatting artifacts
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
                "total_context_chars": total_context_chars
            }
            
        except Exception as e:
            print(f"Error generating answer: {str(e)}")
            return {"success": False, "message": f"Error generating answer: {str(e)}", "answer": None}


# Singleton instance
_rag_service_instance = None

def get_rag_service() -> RAGService:
    """Get singleton RAG service instance"""
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = RAGService()
    return _rag_service_instance