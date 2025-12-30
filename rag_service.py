import os
import numpy as np
import faiss
from typing import List, Dict, Any, Optional
from pymongo import MongoClient
from dotenv import load_dotenv
import google.generativeai as genai
from datetime import datetime
import pickle
import hashlib
import ssl
import pandas as pd
import re

# Load environment variables
load_dotenv()

class RAGService:
    """
    RAG (Retrieval-Augmented Generation) Service using FAISS and Gemini
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
        
        # Indexes storage directory
        self.index_dir = "indexes"
        os.makedirs(self.index_dir, exist_ok=True)
        
        # Configure Gemini
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=self.gemini_api_key)
        
        # Initialize embedding model
        self.embedding_model_name = "models/text-embedding-004"
        self.embedding_dimension = 768
        
        # FAISS indexes per user (in-memory cache)
        self.user_indexes: Dict[str, faiss.IndexIDMap] = {}
        self.user_metadata: Dict[str, Dict[int, Dict[str, Any]]] = {}
    
    def _get_user_index_path(self, user_id: str) -> str:
        return os.path.join(self.index_dir, f"{user_id}_index.faiss")
    
    def _get_user_metadata_path(self, user_id: str) -> str:
        return os.path.join(self.index_dir, f"{user_id}_metadata.pkl")
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a given text using Gemini"""
        try:
            result = genai.embed_content(
                model=self.embedding_model_name,
                content=text,
                task_type="retrieval_document"
            )
            embedding = np.array(result['embedding'], dtype='float32')
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            raise
    
    def _generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a query using Gemini"""
        try:
            result = genai.embed_content(
                model=self.embedding_model_name,
                content=query,
                task_type="retrieval_query"
            )
            embedding = np.array(result['embedding'], dtype='float32')
            return embedding
        except Exception as e:
            print(f"Error generating query embedding: {str(e)}")
            raise
    
    def _create_document_text(self, document: Dict[str, Any]) -> str:
        """Create a searchable text representation from document"""
        parts = []
        
        if document.get('section_title'):
            parts.append(f"Section: {document['section_title']}")
        
        pdf_metadata = document.get('pdf_metadata', {})
        if pdf_metadata.get('title'):
            parts.append(f"Document Title: {pdf_metadata['title']}")
        if pdf_metadata.get('author'):
            parts.append(f"Author: {pdf_metadata['author']}")
        if pdf_metadata.get('subject'):
            parts.append(f"Subject: {pdf_metadata['subject']}")
        
        if document.get('text_content'):
            parts.append(f"Content: {document['text_content']}")
        
        return "\n".join(parts)
    
    def _generate_document_id(self, document: Dict[str, Any]) -> str:
        """Generate a unique ID for a document section"""
        unique_string = f"{document['user_id']}_{document['upload_id']}_{document['section_number']}"
        return hashlib.md5(unique_string.encode()).hexdigest()
    
    def _generate_id_int(self, doc_id: str) -> int:
        """Generate int64 ID from string doc_id"""
        hash_hex = hashlib.sha256(doc_id.encode()).hexdigest()
        return int(hash_hex, 16) & (2**63 - 1)
    
    def _load_text_from_csv(self, csv_path: str, section_number: int) -> Optional[str]:
        """Helper method to load text content from CSV file"""
        try:
            if not os.path.exists(csv_path):
                print(f"CSV file not found: {csv_path}")
                return None
            
            df = pd.read_csv(csv_path, compression='infer')
            section_row = df[df['section_number'] == section_number]
            
            if section_row.empty:
                print(f"Section {section_number} not found in CSV")
                return None
            
            return section_row['text_content'].iloc[0]
        except Exception as e:
            print(f"Error loading text from CSV: {str(e)}")
            return None
    
    def _load_user_index(self, user_id: str) -> bool:
        """
        Load user index from disk or build from scratch if not exists
        FIXED VERSION: Better error handling and index rebuilding
        """
        index_path = self._get_user_index_path(user_id)
        metadata_path = self._get_user_metadata_path(user_id)
        
        # Try to load from disk first
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            try:
                print(f"Loading index from disk for user: {user_id}")
                index = faiss.read_index(index_path)
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                # Verify index is not empty
                if index.ntotal > 0:
                    self.user_indexes[user_id] = index
                    self.user_metadata[user_id] = metadata
                    print(f"Successfully loaded index with {index.ntotal} vectors")
                    return True
                else:
                    print("Loaded index is empty, rebuilding...")
            except Exception as e:
                print(f"Error loading user index, rebuilding: {str(e)}")
        
        # Build from scratch
        print(f"Building index from scratch for user: {user_id}")
        return self._rebuild_user_index(user_id)
    
    def _rebuild_user_index(self, user_id: str) -> bool:
        """
        Rebuild user index from MongoDB documents
        FIXED VERSION: Properly loads text from CSV files
        """
        try:
            # Fetch all documents for this user
            docs = list(self.documents_collection.find({"user_id": user_id}))
            
            print(f"Found {len(docs)} documents for user {user_id}")
            
            if not docs:
                # Create empty index
                self.user_indexes[user_id] = faiss.IndexIDMap(
                    faiss.IndexFlatL2(self.embedding_dimension)
                )
                self.user_metadata[user_id] = {}
                print("No documents found, created empty index")
                return False
            
            # Create new index
            index = faiss.IndexIDMap(faiss.IndexFlatL2(self.embedding_dimension))
            metadata = {}
            
            # Track CSV files we've already loaded
            csv_cache = {}
            
            indexed_count = 0
            for doc in docs:
                try:
                    csv_path = doc.get('csv_path')
                    section_number = doc.get('section_number')
                    
                    if not csv_path or not section_number:
                        print(f"Missing csv_path or section_number for doc {doc.get('_id')}")
                        continue
                    
                    # Load CSV into cache if not already loaded
                    if csv_path not in csv_cache:
                        if not os.path.exists(csv_path):
                            print(f"CSV file not found: {csv_path}")
                            continue
                        csv_cache[csv_path] = pd.read_csv(csv_path, compression='infer')
                    
                    df = csv_cache[csv_path]
                    section_row = df[df['section_number'] == section_number]
                    
                    if section_row.empty:
                        print(f"Section {section_number} not found in CSV")
                        continue
                    
                    text_content = section_row['text_content'].iloc[0]
                    
                    # Create document with text content
                    doc_with_text = {**doc, 'text_content': text_content}
                    searchable_text = self._create_document_text(doc_with_text)
                    
                    # Generate embedding
                    embedding = self._generate_embedding(searchable_text)
                    
                    # Generate IDs
                    doc_id = self._generate_document_id(doc)
                    id_int = self._generate_id_int(doc_id)
                    
                    # Add to index
                    index.add_with_ids(
                        embedding.reshape(1, -1),
                        np.array([id_int], dtype='int64')
                    )
                    
                    # Store metadata
                    metadata[id_int] = {
                        'document_id': doc_id,
                        'upload_id': doc['upload_id'],
                        'section_number': doc['section_number'],
                        'filename': doc['filename'],
                        'section_title': doc['section_title'],
                        'original_doc_ref': doc['_id'],
                        'csv_path': csv_path,
                        'pdf_metadata': doc.get('pdf_metadata', {})
                    }
                    
                    indexed_count += 1
                    
                except Exception as e:
                    print(f"Error processing document {doc.get('_id')}: {str(e)}")
                    continue
            
            print(f"Successfully indexed {indexed_count} documents")
            
            if indexed_count > 0:
                # Save to disk
                faiss.write_index(index, self._get_user_index_path(user_id))
                with open(self._get_user_metadata_path(user_id), 'wb') as f:
                    pickle.dump(metadata, f)
                
                # Store in memory
                self.user_indexes[user_id] = index
                self.user_metadata[user_id] = metadata
                return True
            else:
                # Create empty index
                self.user_indexes[user_id] = faiss.IndexIDMap(
                    faiss.IndexFlatL2(self.embedding_dimension)
                )
                self.user_metadata[user_id] = {}
                return False
            
        except Exception as e:
            print(f"Error rebuilding user index: {str(e)}")
            # Create empty index on error
            self.user_indexes[user_id] = faiss.IndexIDMap(
                faiss.IndexFlatL2(self.embedding_dimension)
            )
            self.user_metadata[user_id] = {}
            return False
    
    def index_document(self, user_id: str, upload_id: str) -> Dict[str, Any]:
        """Index all sections of a newly uploaded document"""
        try:
            # Fetch all sections metadata for this upload
            documents = list(self.documents_collection.find({
                "user_id": user_id,
                "upload_id": upload_id
            }).sort("section_number", 1))
            
            if not documents:
                return {
                    "success": False,
                    "message": "No documents found for this upload_id"
                }
            
            # Load or create index
            self._load_user_index(user_id)
            
            index = self.user_indexes[user_id]
            metadata = self.user_metadata[user_id]
            
            indexed_count = 0
            csv_to_df = {}
            
            for doc in documents:
                csv_path = doc['csv_path']
                if not os.path.exists(csv_path):
                    continue
                
                if csv_path not in csv_to_df:
                    csv_to_df[csv_path] = pd.read_csv(csv_path, compression='infer')
                
                df = csv_to_df[csv_path]
                section_row = df[df['section_number'] == doc['section_number']]
                if section_row.empty:
                    continue
                
                text_content = section_row['text_content'].iloc[0]
                section_type = section_row['section_type'].iloc[0]
                
                doc_with_text = {**doc, 'text_content': text_content, 'section_type': section_type}
                searchable_text = self._create_document_text(doc_with_text)
                embedding = self._generate_embedding(searchable_text)
                
                doc_id = self._generate_document_id(doc)
                id_int = self._generate_id_int(doc_id)
                
                # Remove if exists
                if id_int in metadata:
                    index.remove_ids(np.array([id_int], dtype='int64'))
                    del metadata[id_int]
                
                # Add new
                index.add_with_ids(embedding.reshape(1, -1), np.array([id_int], dtype='int64'))
                metadata[id_int] = {
                    'document_id': doc_id,
                    'upload_id': upload_id,
                    'section_number': doc['section_number'],
                    'filename': doc['filename'],
                    'section_title': doc['section_title'],
                    'original_doc_ref': doc['_id'],
                    'csv_path': csv_path,
                    'pdf_metadata': doc.get('pdf_metadata', {})
                }
                
                indexed_count += 1
            
            # Save after indexing
            faiss.write_index(index, self._get_user_index_path(user_id))
            with open(self._get_user_metadata_path(user_id), 'wb') as f:
                pickle.dump(metadata, f)
            
            return {
                "success": True,
                "indexed_sections": indexed_count,
                "upload_id": upload_id,
                "message": f"Successfully indexed {indexed_count} sections"
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error indexing document: {str(e)}"
            }
    
    def search(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        upload_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search through user's documents using RAG
        FIXED VERSION: Forces index rebuild if empty
        """
        try:
            # Load index (will rebuild if needed)
            if user_id not in self.user_indexes:
                loaded = self._load_user_index(user_id)
                if not loaded:
                    return {
                        "success": False,
                        "message": "No indexed documents found. Please upload and index documents first.",
                        "results": []
                    }
            
            # Get index and verify it's not empty
            index = self.user_indexes[user_id]
            metadata_dict = self.user_metadata[user_id]
            
            if index.ntotal == 0:
                # Try to rebuild
                print(f"Index is empty, attempting rebuild for user {user_id}")
                rebuilt = self._rebuild_user_index(user_id)
                if not rebuilt:
                    return {
                        "success": False,
                        "message": "No indexed documents found. Please upload documents first.",
                        "results": []
                    }
                index = self.user_indexes[user_id]
                metadata_dict = self.user_metadata[user_id]
            
            # Generate query embedding
            query_embedding = self._generate_query_embedding(query)
            query_vector = query_embedding.reshape(1, -1)
            
            # Determine search k
            search_k = min(index.ntotal, top_k * 3) if upload_id else top_k
            
            # Search
            distances, ids = index.search(query_vector, search_k)
            
            # Build results
            results = []
            for i in range(len(ids[0])):
                id_int = ids[0][i]
                if id_int == -1:
                    continue
                meta = metadata_dict.get(id_int)
                if meta:
                    result = meta.copy()
                    result['score'] = float(distances[0][i])
                    result['similarity'] = 1 / (1 + float(distances[0][i]))
                    result['original_doc_ref'] = str(result['original_doc_ref'])
                    results.append(result)
            
            # Filter by upload_id if specified
            if upload_id:
                results = [r for r in results if r['upload_id'] == upload_id]
            
            # Sort by score and take top_k
            results = sorted(results, key=lambda x: x['score'])[:top_k]
            
            return {
                "success": True,
                "query": query,
                "results_count": len(results),
                "results": results
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Search error: {str(e)}",
                "results": []
            }
    
    def delete_document_vectors(self, user_id: str, upload_id: str) -> Dict[str, Any]:
        """Delete all vectors for a specific document"""
        try:
            if user_id not in self.user_indexes:
                self._load_user_index(user_id)
            
            index = self.user_indexes[user_id]
            metadata = self.user_metadata[user_id]
            
            to_remove = []
            for id_int, meta in list(metadata.items()):
                if meta['upload_id'] == upload_id:
                    to_remove.append(id_int)
                    del metadata[id_int]
            
            deleted_count = len(to_remove)
            if to_remove:
                index.remove_ids(np.array(to_remove, dtype='int64'))
                
                # Save
                faiss.write_index(index, self._get_user_index_path(user_id))
                with open(self._get_user_metadata_path(user_id), 'wb') as f:
                    pickle.dump(metadata, f)
            
            return {
                "success": True,
                "deleted_count": deleted_count,
                "message": f"Deleted {deleted_count} vector embeddings"
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error deleting vectors: {str(e)}"
            }
    
    def delete_user_vectors(self, user_id: str) -> Dict[str, Any]:
        """Delete all vectors for a specific user"""
        try:
            index_path = self._get_user_index_path(user_id)
            metadata_path = self._get_user_metadata_path(user_id)
            
            deleted_count = 0
            if os.path.exists(index_path):
                os.remove(index_path)
                deleted_count += 1
            
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            
            if user_id in self.user_indexes:
                del self.user_indexes[user_id]
            if user_id in self.user_metadata:
                del self.user_metadata[user_id]
            
            return {
                "success": True,
                "deleted_count": deleted_count,
                "message": f"Deleted all vectors for user"
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error deleting user vectors: {str(e)}"
            }
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics about user's indexed documents"""
        try:
            self._load_user_index(user_id)
            total_sections = len(self.user_metadata.get(user_id, {}))
            
            # Get unique uploads
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
            return {
                "success": False,
                "message": f"Error getting stats: {str(e)}"
            }
    
    def generate_answer(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        upload_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate an answer to user's query using RAG with Gemini"""
        try:
            # Step 1: Retrieve relevant documents
            search_results = self.search(
                user_id=user_id,
                query=query,
                top_k=top_k,
                upload_id=upload_id
            )
            
            if not search_results["success"]:
                return {
                    "success": False,
                    "message": search_results.get("message", "Search failed"),
                    "answer": None
                }
            
            if not search_results["results"]:
                return {
                    "success": True,
                    "answer": "I couldn't find any relevant information in your documents to answer this question. Please try rephrasing your query or upload relevant documents.",
                    "sources": [],
                    "query": query
                }
            
            # Step 2: Prepare context from retrieved documents
            context_parts = []
            sources = []
            csv_to_df = {}
            
            for i, result in enumerate(search_results["results"], 1):
                csv_path = result['csv_path']
                if csv_path not in csv_to_df:
                    csv_to_df[csv_path] = pd.read_csv(csv_path, compression='infer')
                
                df = csv_to_df[csv_path]
                section_row = df[df['section_number'] == result['section_number']]
                text_content = section_row['text_content'].iloc[0] if not section_row.empty else ""
                
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
            
            # Step 3: Create prompt for Gemini
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
- Keep it flowing and readable"""

            user_prompt = f"""Based on the following excerpts from my documents, please answer my question.

DOCUMENT EXCERPTS:
{context}

MY QUESTION: {query}

Please provide a clear, accurate answer based on the documents above. Reference which documents you're using."""

            # Step 4: Generate response using Gemini
            model = genai.GenerativeModel(
                model_name='gemini-2.0-flash-exp',
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
            return {
                "success": False,
                "message": f"Error generating answer: {str(e)}",
                "answer": None
            }


# Singleton instance
_rag_service_instance = None

def get_rag_service() -> RAGService:
    """Get or create RAG service singleton instance"""
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = RAGService()
    return _rag_service_instance