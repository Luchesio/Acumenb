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

# Load environment variables
load_dotenv()

class RAGService:
    """
    RAG (Retrieval-Augmented Generation) Service using Pinecone and Gemini
    """
    
    def __init__(self):
        # MongoDB configuration (still used for document metadata/CSVs)
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
    
    def _get_user_index_name(self, user_id: str) -> str:
        """Generate a unique, valid index name for each user"""
        # Hash user_id to ensure lowercase alphanum (md5 hex is 0-9a-f)
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
                    region="us-east-1"  # Change to your preferred region
                )
            )
        
        return self.pc.Index(index_name)
    
    def _generate_embedding(self, text: str) -> np.ndarray:
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
        if document.get('section_title'):
            parts.append(f"Section: {document['section_title']}")
        if document.get('pdf_title'):
            parts.append(f"Document Title: {document['pdf_title']}")
        if document.get('pdf_author'):
            parts.append(f"Author: {document['pdf_author']}")
        if document.get('pdf_subject'):
            parts.append(f"Subject: {document['pdf_subject']}")
        if document.get('text_content'):
            parts.append(f"Content: {document['text_content']}")
        return "\n".join(parts)
    
    def _generate_document_id(self, document: Dict[str, Any]) -> str:
        unique_string = f"{document['user_id']}_{document['upload_id']}_{document['section_number']}"
        return hashlib.md5(unique_string.encode()).hexdigest()
    
    def index_document(self, user_id: str, upload_id: str) -> Dict[str, Any]:
        try:
            documents = list(self.documents_collection.find({
                "user_id": user_id,
                "upload_id": upload_id
            }).sort("section_number", 1))
            
            if not documents:
                return {"success": False, "message": "No documents found for this upload_id"}
            
            index = self._ensure_user_index(user_id)
            
            vectors = []
            
            for doc in documents:
                if not doc.get('text_content'):  # Skip if no content
                    continue
                
                searchable_text = self._create_document_text(doc)
                embedding = self._generate_embedding(searchable_text).tolist()  # Pinecone expects list
                
                doc_id = self._generate_document_id(doc)
                
                vectors.append({
                    "id": doc_id,
                    "values": embedding,
                    "metadata": {
                        "upload_id": upload_id,
                        "section_number": doc['section_number'],
                        "filename": doc['filename'],
                        "section_title": doc['section_title'],
                        "original_doc_ref": str(doc['_id']),  # Stringify ObjectId
                        "pdf_author": doc.get('pdf_metadata', {}).get('author', ''),
                        "pdf_title": doc.get('pdf_metadata', {}).get('title', ''),
                        "pdf_subject": doc.get('pdf_metadata', {}).get('subject', '')
                    }
                })
            
            if vectors:
                index.upsert(vectors=vectors, namespace="")  # Use empty string for default namespace
            
            return {
                "success": True,
                "indexed_sections": len(vectors),
                "upload_id": upload_id,
                "message": f"Successfully indexed {len(vectors)} sections in Pinecone"
            }
            
        except Exception as e:
            return {"success": False, "message": f"Error indexing document: {str(e)}"}
    
    def search(self, user_id: str, query: str, top_k: int = 5, upload_id: Optional[str] = None) -> Dict[str, Any]:
        try:
            index = self._ensure_user_index(user_id)
            
            query_embedding = self._generate_query_embedding(query).tolist()
            
            filter = {}
            if upload_id:
                filter["upload_id"] = upload_id
            
            results = index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                namespace="",  # Empty string for default
                filter=filter
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
                    "similarity": match.score,  # Cosine similarity
                    "original_doc_ref": meta['original_doc_ref'],
                    "pdf_author": meta.get('pdf_author', ''),
                    "pdf_title": meta.get('pdf_title', ''),
                    "pdf_subject": meta.get('pdf_subject', '')
                })
            
            return {"success": True, "results": processed_results, "query": query}
            
        except Exception as e:
            return {"success": False, "message": f"Error during search: {str(e)}"}
    
    def delete_document_vectors(self, user_id: str, upload_id: str) -> Dict[str, Any]:
        try:
            index = self._ensure_user_index(user_id)
            
            # Query to get IDs to delete
            results = index.query(
                vector=[0] * self.embedding_dimension,  # Dummy vector
                top_k=10000,  # Adjust based on expected max sections
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
                    "sections_count": {"$sum": 1}
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
    
    def generate_answer(self, user_id: str, query: str, top_k: int = 5, upload_id: Optional[str] = None) -> Dict[str, Any]:
        try:
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
            
            context_parts = []
            sources = []
            
            for i, result in enumerate(search_results["results"], 1):
                # Fetch text_content from MongoDB
                doc = self.documents_collection.find_one({"_id": result["original_doc_ref"]})
                text_content = doc.get('text_content', "") if doc else ""
                
                context_doc = {
                    'section_title': result['section_title'],
                    'pdf_author': result.get('pdf_author', ''),
                    'pdf_title': result.get('pdf_title', ''),
                    'pdf_subject': result.get('pdf_subject', ''),
                    'text_content': text_content
                }
                context_text = self._create_document_text(context_doc)
                
                context_parts.append(
                    f"[Document {i}]\n"
                    f"Source: {result['filename']}\n"
                    f"Section: {result['section_title']}\n"
                    f"Content: {text_content[:1500]}...\n"
                )
                
                sources.append({
                    "filename": result['filename'],
                    "section_title": result['section_title'],
                    "section_number": result['section_number'],
                    "upload_id": result['upload_id'],
                    "similarity": result['similarity']
                })
            
            context = "\n\n".join(context_parts)
            
            system_prompt = """You are a helpful AI assistant answering questions based on PDF documents.

CRITICAL FORMATTING RULES:
- Write in natural paragraphs and flowing prose
- Use **bold text** (double asterisks) ONLY for emphasizing key terms, names, or concepts within sentences
- NEVER use asterisks or dashes for bullet points or lists
- NEVER use numbered lists unless explicitly asked
- Write like you're having a conversation, not writing a report

Guidelines:
- Answer naturally and conversationally
- Base your answer ONLY on the provided documents
- Reference documents inline like: "According to Document 1, ..."
- If information is missing, just say so
- Keep it flowing and readable

Example of good formatting:
"Your documents discuss several key concepts. Document 1 explains how **cognitive bias** affects decision-making, particularly in financial contexts. The author notes that people often create narratives to understand complex situations. Document 2 builds on this by exploring how **identity** forms through repeated behaviors over time."

Stay natural, clear, and conversational."""

            user_prompt = f"""Based on the following excerpts from my documents, please answer my question.

DOCUMENT EXCERPTS:
{context}

MY QUESTION: {query}

Please provide a clear, accurate answer based on the documents above. Reference which documents you're using."""

            model = genai.GenerativeModel(
                model_name='gemini-3-flash-preview',
                system_instruction=system_prompt
            )

            generation_config = genai.GenerationConfig(
                temperature=0.7,
                top_p=0.95,
                top_k=40,
                candidate_count=1,
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
                "context_documents_used": len(sources)
            }
            
        except Exception as e:
            return {"success": False, "message": f"Error generating answer: {str(e)}", "answer": None}


# Singleton instance
_rag_service_instance = None

def get_rag_service() -> RAGService:
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = RAGService()
    return _rag_service_instance