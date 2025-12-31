# Updated main.py with improvements:
# - Integrated pdfplumber for better text extraction
# - Enhanced section detection patterns
# - Improved fallback chunking to paragraph-based

import fitz  # PyMuPDF
import pandas as pd
import re
import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import Optional
import tempfile
import secrets
import hashlib
import ssl
from rag_service import get_rag_service
import csv
import pdfplumber  # New import for better extraction

# Load environment variables
load_dotenv()

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB configuration
MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME")
USERS_COLLECTION = os.getenv("USERS_COLLECTION", "users")
DOCUMENTS_COLLECTION = os.getenv("DOCUMENTS_COLLECTION", "pdf_documents")

# Initialize MongoDB client
context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
context.minimum_version = ssl.TLSVersion.TLSv1_2

client = MongoClient(MONGO_URI, tls=True, tlsAllowInvalidCertificates=True)

db = client[DATABASE_NAME]
users_collection = db[USERS_COLLECTION]
documents_collection = db[DOCUMENTS_COLLECTION]

# Initialize RAG Service
rag_service = get_rag_service()

# Create indexes for better performance
users_collection.create_index("email", unique=True)
users_collection.create_index("user_token")
documents_collection.create_index("user_id")
documents_collection.create_index([("user_id", 1), ("upload_id", 1)])

# --- Helper Functions ---
def generate_user_token():
    """Generate a secure random token for user identification"""
    return secrets.token_urlsafe(32)

def hash_password(password: str) -> str:
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

async def get_current_user(user_token: Optional[str] = Header(None, alias="X-User-Token")):
    """Dependency to get current user from token"""
    if not user_token:
        raise HTTPException(status_code=401, detail="User token is required")
    
    user = users_collection.find_one({"user_token": user_token})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid user token")
    
    return user

# --- PDF Processing Functions ---
def clean_text(text: str) -> str:
    """Clean extracted text"""
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

def extract_sections_advanced(doc, pdf_path: str):  # Added pdf_path param for pdfplumber
    """
    Advanced section detection for any PDF type.
    Handles documents with or without clear chapter structure.
    """
    full_text = ""
    page_texts = []
    
    # Use pdfplumber for better extraction
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_text = page.extract_text(layout=True) or ""
            if not page_text.strip():  # Fallback to table extraction if no text
                tables = page.extract_tables()
                if tables:
                    page_text = "\n".join([" | ".join(row) for table in tables for row in table])
            
            page_texts.append({
                'page_number': page_num + 1,
                'text': page_text
            })
            full_text += page_text + "\n"
    
    # Enhanced patterns
    patterns = [
        r'^(?:Chapter|CHAPTER|Ch\.|Section|SECTION|Sec\.|Part|PART)\s+(\d+|[IVXLCDM]+)[:\.\s-]*(.+?)$',  # More variations
        r'^(?:Section|SECTION)\s+(\d+|[IVXLCDM]+)[:\.\s]+(.+?)$',  # Section 1: Title
        r'^\s*(\d+\.\d+|\d+)(?:\.\s*|\s+)([A-Z0-9][^\n]{5,100})$',  # Numbered sections with sublevels
        r'^([A-Z][A-Z0-9\s&-]{5,100})$',  # ALL CAPS or bold-like headings (len >5 to avoid noise)
        r'^\s*([IVXLCDM]+)\.\s+(.+?)$',  # Roman numerals
        r'^(?:Article|ARTICLE)\s+(\d+|[IVXLCDM]+)[:\.\s]+(.+?)$',  # Article patterns
        r'^(\d+)\.\s+([A-Z][^\n]{10,100})$',  # 1. Title Format
        r'^([A-Z][A-Z\s]{3,50})$',  # ALL CAPS HEADINGS
        r'^(\d+\.\d+)\s+(.+?)$',  # 1.1 Subsection
    ]
    
    sections_data = []
    detected_sections = []
    
    # Try to detect sections using patterns
    for pattern in patterns:
        matches = list(re.finditer(pattern, full_text, re.MULTILINE))
        if len(matches) >= 2:  # At least 2 sections found
            detected_sections = matches
            break
    
    if detected_sections:
        # Process detected sections
        for i, match in enumerate(detected_sections):
            section_title = match.group(0).strip()
            start_pos = match.start()
            
            # Get text until next section or end
            if i + 1 < len(detected_sections):
                end_pos = detected_sections[i + 1].start()
            else:
                end_pos = len(full_text)
            
            section_text = full_text[start_pos:end_pos].strip()
            
            # Estimate page numbers
            chars_before = len(full_text[:start_pos])
            chars_per_page = len(full_text) / len(doc) if len(doc) > 0 else 1000
            start_page = max(1, int(chars_before / chars_per_page) + 1)
            
            chars_in_section = len(section_text)
            pages_in_section = max(1, int(chars_in_section / chars_per_page))
            end_page = min(len(doc), start_page + pages_in_section)
            
            sections_data.append({
                'section_title': section_title,
                'start_page': start_page,
                'end_page': end_page,
                'text_content': clean_text(section_text),
                'section_type': 'detected'
            })
    else:
        # No clear sections detected - split by pages or create logical chunks
        if len(doc) <= 5:
            # Short document - treat as single section
            sections_data.append({
                'section_title': 'Complete Document',
                'start_page': 1,
                'end_page': len(doc),
                'text_content': clean_text(full_text),
                'section_type': 'full_document'
            })
        else:
            # Improved fallback: Split into paragraph-based chunks
            paragraphs = re.split(r'\n\s*\n', full_text.strip())  # Split on double newlines
            current_page = 1
            chunk_text = ""
            for para in paragraphs:
                if len(chunk_text) + len(para) > 2000:  # ~500 words per chunk
                    # Find a title-like line or use pages
                    first_line = para.split('\n')[0].strip()
                    chunk_title = first_line if len(first_line) > 10 else f"Chunk starting page {current_page}"
                    
                    sections_data.append({
                        'section_title': chunk_title,
                        'start_page': current_page,
                        'end_page': current_page,  # Approximate; improve if needed
                        'text_content': clean_text(chunk_text),
                        'section_type': 'paragraph_chunk'
                    })
                    chunk_text = para
                    current_page += 1  # Rough estimate
                else:
                    chunk_text += "\n\n" + para
            
            # Add last chunk
            if chunk_text:
                chunk_title = chunk_text.split('\n')[0].strip() or f"Final Chunk"
                sections_data.append({
                    'section_title': chunk_title,
                    'start_page': current_page,
                    'end_page': len(doc),
                    'text_content': clean_text(chunk_text),
                    'section_type': 'paragraph_chunk'
                })
    
    return sections_data

def process_pdf_to_mongodb(pdf_path: str, user_id: str, filename: str):
    """Process PDF and insert sections into MongoDB"""
    try:
        doc = fitz.open(pdf_path)
        
        # Extract sections using advanced detection (pass pdf_path for pdfplumber)
        sections_data = extract_sections_advanced(doc, pdf_path)
        
        # Get PDF metadata
        metadata = doc.metadata
        total_pages = len(doc)
        
        doc.close()
        
        if not sections_data:
            raise Exception("No content extracted from PDF")
        
        # Generate unique upload ID
        upload_id = secrets.token_urlsafe(16)
        
        documents = []
        
        for idx, section in enumerate(sections_data):
            section_number = idx + 1
            
            document = {
                "user_id": user_id,
                "upload_id": upload_id,
                "filename": filename,
                "section_number": section_number,
                "section_title": section['section_title'],
                "section_type": section['section_type'],
                "text_content": section['text_content'],  # Store directly in MongoDB
                "start_page": section['start_page'],
                "end_page": section['end_page'],
                "total_pages": total_pages,
                "pdf_metadata": {
                    "author": metadata.get("author", ""),
                    "title": metadata.get("title", ""),
                    "subject": metadata.get("subject", ""),
                },
                "uploaded_at": datetime.utcnow()
            }
            documents.append(document)
        
        # Insert into MongoDB
        result = documents_collection.insert_many(documents)
        
        return {
            "success": True,
            "upload_id": upload_id,
            "sections_inserted": len(result.inserted_ids),
            "total_pages": total_pages,
            "message": "PDF processed and saved to database successfully"
        }
        
    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}")

# --- API Endpoints ---
@app.get("/")
async def root():
    return {
        "message": "PDF Processing API with RAG Search", 
        "version": "3.0",
        "features": ["User Authentication", "PDF Processing", "RAG Search with Pinecone + Gemini"]
    }

@app.post("/register")
async def register_user(email: str, password: str, name: str):
    """Register a new user"""
    try:
        # Check if user already exists
        existing_user = users_collection.find_one({"email": email.lower()})
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Create user
        user_token = generate_user_token()
        user_id = secrets.token_urlsafe(16)
        
        user_doc = {
            "user_id": user_id,
            "email": email.lower(),
            "name": name,
            "password_hash": hash_password(password),
            "user_token": user_token,
            "created_at": datetime.utcnow(),
            "last_login": datetime.utcnow()
        }
        
        users_collection.insert_one(user_doc)
        
        return {
            "success": True,
            "user_token": user_token,
            "user_id": user_id,
            "name": name,
            "email": email,
            "message": "User registered successfully"
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/document/{upload_id}")
async def delete_document(
    upload_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a document and all its sections"""
    try:
        # Delete from documents collection
        result = documents_collection.delete_many({
            "user_id": current_user["user_id"],
            "upload_id": upload_id
        })
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete vector embeddings
        rag_result = rag_service.delete_document_vectors(
            current_user["user_id"],
            upload_id
        )
        
        return {
            "success": True,
            "deleted_sections": result.deleted_count,
            "rag_deletion": rag_result,
            "message": "Document deleted successfully"
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rag-search")
async def rag_search(
    query: str,
    top_k: int = 5,
    upload_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """
    RAG-powered semantic search through user's documents
    
    Args:
        query: Search query text
        top_k: Number of results to return (default: 5)
        upload_id: Optional - limit search to specific document
    """
    try:
        if not query or len(query.strip()) < 3:
            raise HTTPException(
                status_code=400, 
                detail="Query must be at least 3 characters"
            )
        
        # Perform RAG search
        results = rag_service.search(
            user_id=current_user["user_id"],
            query=query,
            top_k=top_k,
            upload_id=upload_id
        )
        
        if not results["success"]:
            return results
        
        # Enhance results with full document details
        enhanced_results = []
        for result in results["results"]:
            # Fetch metadata from MongoDB
            doc = documents_collection.find_one(
                {"_id": result["original_doc_ref"]},
                {"pdf_metadata": 1, "start_page": 1, "end_page": 1, "text_content": 1, "section_number": 1}
            )
            
            if doc:
                result["pdf_metadata"] = doc.get("pdf_metadata", {})
                result["start_page"] = doc.get("start_page")
                result["end_page"] = doc.get("end_page")
                result["text_content"] = doc.get("text_content", "")
            
            enhanced_results.append(result)
        
        results["results"] = enhanced_results
        
        return results
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
async def search_documents(
    query: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Legacy text search (keeping for backward compatibility)
    Redirects to RAG search
    """
    return await rag_search(query=query, top_k=10, current_user=current_user)

@app.get("/rag-stats")
async def get_rag_stats(current_user: dict = Depends(get_current_user)):
    """Get RAG indexing statistics for current user"""
    try:
        stats = rag_service.get_user_stats(current_user["user_id"])
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reindex-document/{upload_id}")
async def reindex_document(
    upload_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Reindex a specific document for RAG search"""
    try:
        # Verify document exists
        doc_exists = documents_collection.find_one({
            "user_id": current_user["user_id"],
            "upload_id": upload_id
        })
        
        if not doc_exists:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Reindex
        result = rag_service.index_document(
            current_user["user_id"],
            upload_id
        )
        
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reindex-all")
async def reindex_all_documents(current_user: dict = Depends(get_current_user)):
    """Reindex all documents for current user"""
    try:
        # Get all unique upload_ids for this user
        pipeline = [
            {"$match": {"user_id": current_user["user_id"]}},
            {"$group": {"_id": "$upload_id"}}
        ]
        
        upload_ids = [doc["_id"] for doc in documents_collection.aggregate(pipeline)]
        
        if not upload_ids:
            return {
                "success": True,
                "message": "No documents to index",
                "indexed_documents": 0
            }
        
        indexed_count = 0
        errors = []
        
        for upload_id in upload_ids:
            result = rag_service.index_document(
                current_user["user_id"],
                upload_id
            )
            
            if result["success"]:
                indexed_count += 1
            else:
                errors.append({
                    "upload_id": upload_id,
                    "error": result.get("message", "Unknown error")
                })
        
        return {
            "success": True,
            "total_documents": len(upload_ids),
            "indexed_documents": indexed_count,
            "failed_documents": len(errors),
            "errors": errors if errors else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(
    query: str,
    top_k: int = 5,
    upload_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """
    Ask a question and get an AI-generated answer based on your documents using RAG
    
    This endpoint:
    1. Searches your documents using semantic search (FAISS + Gemini embeddings)
    2. Retrieves the most relevant sections
    3. Uses Gemini 2.5 Flash to generate a comprehensive answer
    4. Returns the answer with source citations
    
    Args:
        query: Your question (e.g., "What are the main findings in chapter 3?")
        top_k: Number of document sections to use as context (default: 5)
        upload_id: Optional - limit the search to a specific document
    
    Returns:
        - answer: AI-generated response based on your documents
        - sources: List of document sections used to generate the answer
        - query: Your original question
    """
    try:
        if not query or len(query.strip()) < 3:
            raise HTTPException(
                status_code=400,
                detail="Question must be at least 3 characters"
            )
        
        # Generate answer using RAG
        result = rag_service.generate_answer(
            user_id=current_user["user_id"],
            query=query,
            top_k=top_k,
            upload_id=upload_id
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=result.get("message", "Failed to generate answer")
            )
        
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

@app.post("/login")
async def login_user(email: str, password: str):
    """Login user and return token"""
    try:
        user = users_collection.find_one({"email": email.lower()})
        
        if not user or user["password_hash"] != hash_password(password):
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        # Update last login
        users_collection.update_one(
            {"email": email.lower()},
            {"$set": {"last_login": datetime.utcnow()}}
        )
        
        return {
            "success": True,
            "user_token": user["user_token"],
            "user_id": user["user_id"],
            "name": user["name"],
            "email": user["email"],
            "message": "Login successful"
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Upload and process PDF"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Process the PDF
        result = process_pdf_to_mongodb(
            tmp_file_path, 
            current_user["user_id"], 
            file.filename
        )
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        if result["success"]:
            # Index the document for RAG search
            index_result = rag_service.index_document(
                current_user["user_id"],
                result["upload_id"]
            )
            
            result["rag_indexing"] = index_result
        
        return result
        
    except Exception as e:
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/my-documents")
async def get_my_documents(current_user: dict = Depends(get_current_user)):
    """Get all documents for current user"""
    try:
        # Aggregate documents by upload_id
        pipeline = [
            {"$match": {"user_id": current_user["user_id"]}},
            {"$group": {
                "_id": "$upload_id",
                "filename": {"$first": "$filename"},
                "uploaded_at": {"$first": "$uploaded_at"},
                "total_sections": {"$sum": 1},
                "total_pages": {"$first": "$total_pages"},
                "pdf_title": {"$first": "$pdf_metadata.title"}
            }},
            {"$sort": {"uploaded_at": -1}}
        ]
        
        documents = list(documents_collection.aggregate(pipeline))
        
        return {
            "success": True,
            "user_id": current_user["user_id"],
            "total_uploads": len(documents),
            "documents": documents
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/document/{upload_id}")
async def get_document_details(
    upload_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get detailed sections of a specific document"""
    try:
        sections = list(documents_collection.find(
            {
                "user_id": current_user["user_id"],
                "upload_id": upload_id
            }
        ).sort("section_number", 1))
        
        if not sections:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "success": True,
            "upload_id": upload_id,
            "filename": sections[0]["filename"],
            "total_sections": len(sections),
            "sections": sections
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))