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

# --- IMPROVED PDF PROCESSING FUNCTIONS ---
def clean_text(text: str) -> str:
    """
    Enhanced text cleaning that preserves content while removing artifacts
    """
    if not text:
        return ""
    
    # Remove null bytes and special characters that cause issues
    text = text.replace('\x00', '')
    
    # Normalize unicode characters
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    
    # Remove excessive whitespace while preserving paragraph breaks
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Max 2 newlines
    text = re.sub(r' +', ' ', text)  # Multiple spaces to single
    text = re.sub(r'\t+', ' ', text)  # Tabs to spaces
    
    # Remove page numbers and headers/footers patterns (common artifacts)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*Page \d+\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    return text.strip()

def extract_text_from_page(page) -> str:
    """
    Enhanced text extraction from a single page using multiple methods
    """
    # Method 1: Standard text extraction
    text = page.get_text("text")
    
    # Method 2: If standard extraction fails or returns very little, try blocks
    if not text or len(text.strip()) < 50:
        try:
            blocks = page.get_text("blocks")
            text_parts = []
            for block in blocks:
                if len(block) >= 5 and block[4].strip():  # block[4] contains text
                    text_parts.append(block[4])
            text = "\n".join(text_parts)
        except:
            pass
    
    # Method 3: If still no content, try dict method (most detailed)
    if not text or len(text.strip()) < 50:
        try:
            text_dict = page.get_text("dict")
            text_parts = []
            for block in text_dict.get("blocks", []):
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        line_text = ""
                        for span in line.get("spans", []):
                            line_text += span.get("text", "")
                        if line_text.strip():
                            text_parts.append(line_text)
            text = "\n".join(text_parts)
        except:
            pass
    
    return clean_text(text)

def extract_sections_advanced(doc):
    """
    IMPROVED: Enhanced section detection with full content extraction
    """
    full_text = ""
    page_texts = []
    
    # Extract text from each page using enhanced method
    print(f"Extracting text from {len(doc)} pages...")
    for page_num, page in enumerate(doc):
        page_text = extract_text_from_page(page)
        
        if page_text:  # Only add pages with actual content
            page_texts.append({
                'page_number': page_num + 1,
                'text': page_text,
                'char_count': len(page_text)
            })
            full_text += page_text + "\n\n"
        else:
            print(f"Warning: Page {page_num + 1} contains no extractable text")
    
    print(f"Total extracted text length: {len(full_text)} characters")
    
    if not full_text.strip():
        raise Exception("No text content could be extracted from PDF. The PDF might be image-based or encrypted.")
    
    # Enhanced section detection patterns
    patterns = [
        # Chapter patterns
        r'^(?:Chapter|CHAPTER|Ch\.|CH\.)\s+(\d+|[IVXLCDM]+)[\:\.\s]+(.+?)$',
        # Section patterns
        r'^(?:Section|SECTION|Sec\.|SEC\.)\s+(\d+|[IVXLCDM]+)[\:\.\s]+(.+?)$',
        # Numbered sections
        r'^(\d+)\.\s+([A-Z][^\n]{10,100})$',
        # All caps headings (minimum 4 chars, max 80)
        r'^([A-Z][A-Z\s]{3,80})$',
        # Decimal sections
        r'^(\d+\.\d+)\s+(.+?)$',
        # Part/Book patterns
        r'^(?:Part|PART|Book|BOOK)\s+(\d+|[IVXLCDM]+)[\:\.\s]+(.+?)$',
    ]
    
    sections_data = []
    detected_sections = []
    
    # Try to detect sections using patterns
    for pattern_idx, pattern in enumerate(patterns):
        matches = list(re.finditer(pattern, full_text, re.MULTILINE))
        if len(matches) >= 2:  # At least 2 sections found
            detected_sections = matches
            print(f"Detected {len(matches)} sections using pattern {pattern_idx + 1}")
            break
    
    if detected_sections and len(detected_sections) >= 2:
        # Process detected sections - FULL CONTENT, NO TRUNCATION
        for i, match in enumerate(detected_sections):
            section_title = match.group(0).strip()
            start_pos = match.start()
            
            # Get FULL text until next section or end
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
                'text_content': section_text,  # FULL CONTENT
                'section_type': 'detected',
                'char_count': len(section_text)
            })
            
            print(f"Section {i+1}: '{section_title[:50]}...' - {len(section_text)} chars")
    
    else:
        # No clear sections - intelligent chunking based on document size
        print("No clear sections detected. Using intelligent chunking...")
        
        if len(doc) <= 5:
            # Short document - treat as single section
            sections_data.append({
                'section_title': 'Complete Document',
                'start_page': 1,
                'end_page': len(doc),
                'text_content': full_text,  # ENTIRE document
                'section_type': 'full_document',
                'char_count': len(full_text)
            })
            print(f"Single section created with {len(full_text)} characters")
        
        else:
            # Larger document - smart chunking by pages (5-8 pages per chunk)
            # This ensures each chunk is substantial but not too large
            chunk_size = 6  # Increased from 3 for better context
            
            for i in range(0, len(page_texts), chunk_size):
                chunk_pages = page_texts[i:i + chunk_size]
                start_page = chunk_pages[0]['page_number']
                end_page = chunk_pages[-1]['page_number']
                
                # Combine FULL text from all pages in chunk
                chunk_text = "\n\n".join([p['text'] for p in chunk_pages])
                
                # Try to find a meaningful heading in first 10 lines
                first_lines = chunk_text.split('\n')[:10]
                chunk_title = None
                
                for line in first_lines:
                    line = line.strip()
                    # Look for lines that could be headings
                    if 10 <= len(line) <= 100 and not line.endswith('.'):
                        chunk_title = line
                        break
                
                if not chunk_title:
                    chunk_title = f"Pages {start_page}-{end_page}"
                
                sections_data.append({
                    'section_title': chunk_title,
                    'start_page': start_page,
                    'end_page': end_page,
                    'text_content': chunk_text,  # FULL CHUNK CONTENT
                    'section_type': 'page_chunk',
                    'char_count': len(chunk_text)
                })
                
                print(f"Chunk {len(sections_data)}: Pages {start_page}-{end_page} - {len(chunk_text)} chars")
    
    # Verify we captured all content
    total_extracted = sum(s['char_count'] for s in sections_data)
    print(f"\nExtraction Summary:")
    print(f"- Total sections: {len(sections_data)}")
    print(f"- Original text: {len(full_text)} chars")
    print(f"- Extracted text: {total_extracted} chars")
    print(f"- Coverage: {(total_extracted/len(full_text)*100):.1f}%")
    
    return sections_data

def process_pdf_to_mongodb(pdf_path: str, user_id: str, filename: str):
    """Process PDF and insert sections into MongoDB with FULL content"""
    try:
        doc = fitz.open(pdf_path)
        
        print(f"\n{'='*60}")
        print(f"Processing PDF: {filename}")
        print(f"Total pages: {len(doc)}")
        print(f"{'='*60}\n")
        
        # Extract sections using improved detection
        sections_data = extract_sections_advanced(doc)
        
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
                "text_content": section['text_content'],  # FULL CONTENT - NO TRUNCATION
                "char_count": section['char_count'],  # Track content size
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
        
        print(f"\n{'='*60}")
        print(f"Successfully saved {len(result.inserted_ids)} sections to MongoDB")
        print(f"{'='*60}\n")
        
        return {
            "success": True,
            "upload_id": upload_id,
            "sections_inserted": len(result.inserted_ids),
            "total_pages": total_pages,
            "total_characters": sum(s['char_count'] for s in sections_data),
            "message": "PDF processed and saved to database successfully"
        }
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        raise Exception(f"Error processing PDF: {str(e)}")

# --- API ENDPOINTS (keeping your existing endpoints) ---
@app.get("/")
async def root():
    return {
        "message": "PDF Processing API with RAG Search", 
        "version": "3.1-IMPROVED",
        "improvements": ["Full content extraction", "Multi-method text extraction", "Better section detection"],
        "features": ["User Authentication", "PDF Processing", "RAG Search with Pinecone + Gemini"]
    }

@app.post("/register")
async def register_user(email: str, password: str, name: str):
    """Register a new user"""
    try:
        existing_user = users_collection.find_one({"email": email.lower()})
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")
        
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

@app.post("/login")
async def login_user(email: str, password: str):
    """Login user and return token"""
    try:
        user = users_collection.find_one({"email": email.lower()})
        
        if not user or user["password_hash"] != hash_password(password):
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
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
    """Upload and process PDF with improved extraction"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        result = process_pdf_to_mongodb(
            tmp_file_path, 
            current_user["user_id"], 
            file.filename
        )
        
        os.unlink(tmp_file_path)
        
        if result["success"]:
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

# ... (keep all your other endpoints: my-documents, document/{upload_id}, 
# delete, rag-search, ask, reindex, etc.)

@app.get("/my-documents")
async def get_my_documents(current_user: dict = Depends(get_current_user)):
    """Get all documents for current user"""
    try:
        pipeline = [
            {"$match": {"user_id": current_user["user_id"]}},
            {"$group": {
                "_id": "$upload_id",
                "filename": {"$first": "$filename"},
                "uploaded_at": {"$first": "$uploaded_at"},
                "total_sections": {"$sum": 1},
                "total_pages": {"$first": "$total_pages"},
                "total_characters": {"$sum": "$char_count"},  # New field
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
            "total_characters": sum(s.get("char_count", 0) for s in sections),
            "sections": sections
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
        result = documents_collection.delete_many({
            "user_id": current_user["user_id"],
            "upload_id": upload_id
        })
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Document not found")
        
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
    """RAG-powered semantic search through user's documents"""
    try:
        if not query or len(query.strip()) < 3:
            raise HTTPException(status_code=400, detail="Query must be at least 3 characters")
        
        results = rag_service.search(
            user_id=current_user["user_id"],
            query=query,
            top_k=top_k,
            upload_id=upload_id
        )
        
        if not results["success"]:
            return results
        
        enhanced_results = []
        for result in results["results"]:
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

@app.post("/ask")
async def ask_question(
    query: str,
    top_k: int = 5,
    upload_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Ask a question and get an AI-generated answer based on your documents using RAG."""
    try:
        # FIX: lowered minimum from 3 chars → 1 char so greetings like "hi"
        # are accepted and handled by the conversational-intent layer in RAGService.
        if not query or len(query.strip()) < 1:
            raise HTTPException(status_code=400, detail="Please enter a message.")
 
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
        doc_exists = documents_collection.find_one({
            "user_id": current_user["user_id"],
            "upload_id": upload_id
        })
        
        if not doc_exists:
            raise HTTPException(status_code=404, detail="Document not found")
        
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)