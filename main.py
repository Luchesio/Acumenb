import fitz  # PyMuPDF
import re
import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import tempfile
import secrets
import hashlib
import ssl
from rag_service import get_rag_service
import bcrypt
from jose import JWTError, jwt
from pydantic import BaseModel, field_validator

# Load environment variables
load_dotenv()

# ─── Chunking Configuration ──────────────────────────────────────────────────
CHUNK_SIZE    = 4000   # target characters per chunk in fallback mode
CHUNK_OVERLAP = 400    # overlap between consecutive chunks
MIN_CHUNK_SIZE = 200   # chunks smaller than this are discarded


def split_text_with_overlap(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    min_chunk_size: int = MIN_CHUNK_SIZE,
) -> list[str]:
    """
    Split text into overlapping chunks that respect natural language boundaries.
    """
    if not text or len(text.strip()) < min_chunk_size:
        return []

    text = text.strip()

    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end == len(text):
            chunk = text[start:].strip()
            if len(chunk) >= min_chunk_size:
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
        if len(chunk) >= min_chunk_size:
            chunks.append(chunk)

        start = max(start + 1, break_point - chunk_overlap)

        snap_end = min(start + chunk_overlap // 2, len(text))
        for sep in ['\n\n', '\n', '. ']:
            bp = text.find(sep, start, snap_end)
            if bp != -1:
                start = bp + len(sep)
                break

    return chunks


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

# CHANGED: pdf_uploads stores only lightweight metadata — NO text_content.
# The full text lives exclusively in Pinecone vector metadata.
PDF_UPLOADS_COLLECTION = os.getenv("PDF_UPLOADS_COLLECTION", "pdf_uploads")

# JWT configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not JWT_SECRET_KEY:
    JWT_SECRET_KEY = secrets.token_urlsafe(64)
    print("WARNING: JWT_SECRET_KEY not set in environment. Using ephemeral key — all tokens will be invalidated on restart.")

JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 30

# Initialize MongoDB client
context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
context.minimum_version = ssl.TLSVersion.TLSv1_2

client = MongoClient(MONGO_URI, tls=True, tlsAllowInvalidCertificates=True)

db = client[DATABASE_NAME]
users_collection = db[USERS_COLLECTION]
pdf_uploads_collection = db[PDF_UPLOADS_COLLECTION]   # lightweight metadata only

# Initialize RAG Service
rag_service = get_rag_service()

# Create indexes
users_collection.create_index("email", unique=True)
users_collection.create_index("user_id", unique=True)
pdf_uploads_collection.create_index("user_id")
pdf_uploads_collection.create_index([("user_id", 1), ("upload_id", 1)], unique=True)


# ─── Pydantic Request Models ─────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def email_must_be_valid(cls, v: str) -> str:
        v = v.strip().lower()
        pattern = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
        if not re.match(pattern, v):
            raise ValueError("Invalid email address")
        return v

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError("Password must be at least 6 characters")
        return v

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Name cannot be empty")
        if len(v) > 100:
            raise ValueError("Name is too long")
        return v


class LoginRequest(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def email_normalise(cls, v: str) -> str:
        return v.strip().lower()


# ─── Password Helpers ─────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, stored_hash: str, user_email: str) -> bool:
    if len(stored_hash) == 64 and re.fullmatch(r"[0-9a-f]{64}", stored_hash):
        old_hash = hashlib.sha256(plain_password.encode()).hexdigest()
        if old_hash != stored_hash:
            return False
        new_hash = hash_password(plain_password)
        users_collection.update_one(
            {"email": user_email},
            {"$set": {"password_hash": new_hash}}
        )
        return True

    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), stored_hash.encode("utf-8"))
    except Exception:
        return False


# ─── JWT Helpers ──────────────────────────────────────────────────────────────

def create_access_token(user_id: str, email: str) -> str:
    expire = datetime.utcnow() + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    payload = {
        "sub": user_id,
        "email": email,
        "exp": expire,
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


async def get_current_user(
    user_token: Optional[str] = Header(None, alias="X-User-Token")
):
    if not user_token:
        raise HTTPException(status_code=401, detail="Authentication required. Please log in.")

    try:
        payload = jwt.decode(user_token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id: Optional[str] = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token payload.")
    except JWTError as exc:
        error_detail = "Your session has expired. Please log in again."
        if "Signature" in str(exc) or "invalid" in str(exc).lower():
            error_detail = "Invalid token. Please log in again."
        raise HTTPException(status_code=401, detail=error_detail)

    user = users_collection.find_one({"user_id": user_id})
    if not user:
        raise HTTPException(status_code=401, detail="User account not found.")

    return user


# ─── PDF Processing ───────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace('\x00', '')
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\t+', ' ', text)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*Page \d+\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    return text.strip()


def extract_text_from_page(page) -> str:
    text = page.get_text("text")
    if not text or len(text.strip()) < 50:
        try:
            blocks = page.get_text("blocks")
            text_parts = [block[4] for block in blocks if len(block) >= 5 and block[4].strip()]
            text = "\n".join(text_parts)
        except Exception:
            pass
    if not text or len(text.strip()) < 50:
        try:
            text_dict = page.get_text("dict")
            text_parts = []
            for block in text_dict.get("blocks", []):
                if block.get("type") == 0:
                    for line in block.get("lines", []):
                        line_text = "".join(span.get("text", "") for span in line.get("spans", []))
                        if line_text.strip():
                            text_parts.append(line_text)
            text = "\n".join(text_parts)
        except Exception:
            pass
    return clean_text(text)


def extract_sections_advanced(doc) -> List[Dict[str, Any]]:
    """Extract and chunk PDF text into sections. Returns list of section dicts."""
    full_text = ""
    page_texts = []

    print(f"Extracting text from {len(doc)} pages...")
    for page_num, page in enumerate(doc):
        page_text = extract_text_from_page(page)
        if page_text:
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

    patterns = [
        r'^(?:Chapter|CHAPTER|Ch\.|CH\.)\s+(\d+|[IVXLCDM]+)[\:\.\s]+(.+?)$',
        r'^(?:Section|SECTION|Sec\.|SEC\.)\s+(\d+|[IVXLCDM]+)[\:\.\s]+(.+?)$',
        r'^(\d+)\.\s+([A-Z][^\n]{10,100})$',
        r'^([A-Z][A-Z\s]{3,80})$',
        r'^(\d+\.\d+)\s+(.+?)$',
        r'^(?:Part|PART|Book|BOOK)\s+(\d+|[IVXLCDM]+)[\:\.\s]+(.+?)$',
    ]

    sections_data = []
    detected_sections = []

    for pattern_idx, pattern in enumerate(patterns):
        matches = list(re.finditer(pattern, full_text, re.MULTILINE))
        if len(matches) >= 2:
            detected_sections = matches
            print(f"Detected {len(matches)} sections using pattern {pattern_idx + 1}")
            break

    chars_per_page = len(full_text) / len(doc) if len(doc) > 0 else 1000

    if detected_sections and len(detected_sections) >= 2:
        for i, match in enumerate(detected_sections):
            section_title = match.group(0).strip()
            start_pos = match.start()
            end_pos = (
                detected_sections[i + 1].start()
                if i + 1 < len(detected_sections)
                else len(full_text)
            )
            section_text = full_text[start_pos:end_pos].strip()

            if len(section_text) < MIN_CHUNK_SIZE:
                print(f"  Skipping tiny section '{section_title}' ({len(section_text)} chars)")
                continue

            chars_before = len(full_text[:start_pos])
            start_page = max(1, int(chars_before / chars_per_page) + 1)
            pages_in_section = max(1, int(len(section_text) / chars_per_page))
            end_page = min(len(doc), start_page + pages_in_section)

            if len(section_text) > CHUNK_SIZE:
                sub_chunks = split_text_with_overlap(section_text)
                for sub_idx, sub_text in enumerate(sub_chunks, start=1):
                    sub_title = (
                        f"{section_title} (part {sub_idx}/{len(sub_chunks)})"
                        if len(sub_chunks) > 1
                        else section_title
                    )
                    sections_data.append({
                        'section_title': sub_title,
                        'start_page': start_page,
                        'end_page': end_page,
                        'text_content': sub_text,
                        'section_type': 'detected_sub',
                        'char_count': len(sub_text)
                    })
            else:
                sections_data.append({
                    'section_title': section_title,
                    'start_page': start_page,
                    'end_page': end_page,
                    'text_content': section_text,
                    'section_type': 'detected',
                    'char_count': len(section_text)
                })
    else:
        print("No clear sections detected. Using overlap-aware character chunking...")

        if len(doc) <= 5:
            if len(full_text) >= MIN_CHUNK_SIZE:
                sections_data.append({
                    'section_title': 'Complete Document',
                    'start_page': 1,
                    'end_page': len(doc),
                    'text_content': full_text,
                    'section_type': 'full_document',
                    'char_count': len(full_text)
                })
        else:
            text_chunks = split_text_with_overlap(full_text)
            cumulative = 0
            for chunk_text in text_chunks:
                start_page = max(1, int(cumulative / chars_per_page) + 1)
                end_page = min(
                    len(doc),
                    int((cumulative + len(chunk_text)) / chars_per_page) + 1
                )

                chunk_title = None
                for line in chunk_text.split('\n')[:10]:
                    line = line.strip()
                    if 10 <= len(line) <= 100 and not line.endswith('.'):
                        chunk_title = line
                        break
                if not chunk_title:
                    chunk_title = f"Pages {start_page}–{end_page}"

                if len(chunk_text) < MIN_CHUNK_SIZE:
                    print(f"  Skipping tiny fallback chunk ({len(chunk_text)} chars)")
                    cumulative += len(chunk_text)
                    continue

                sections_data.append({
                    'section_title': chunk_title,
                    'start_page': start_page,
                    'end_page': end_page,
                    'text_content': chunk_text,
                    'section_type': 'overlap_chunk',
                    'char_count': len(chunk_text)
                })

                cumulative += max(1, len(chunk_text) - CHUNK_OVERLAP)

    total_extracted = sum(s['char_count'] for s in sections_data)
    print(f"\nExtraction Summary:")
    print(f"- Total sections: {len(sections_data)}")
    print(f"- Original text: {len(full_text)} chars")
    print(f"- Extracted text: {total_extracted} chars")
    print(f"- Coverage: {(total_extracted/len(full_text)*100):.1f}%")

    return sections_data


def process_pdf(pdf_path: str, user_id: str, filename: str):
    """
    CHANGED: Extract sections from PDF and return them in-memory.
    Nothing is stored in MongoDB here. The caller is responsible for:
      1. Saving lightweight metadata to pdf_uploads_collection
      2. Passing sections to rag_service.index_document()

    Returns:
        (sections, metadata_doc)
        sections      - list of section dicts (with text_content) for Pinecone indexing
        metadata_doc  - lightweight dict (no text_content) to store in MongoDB
    """
    doc = fitz.open(pdf_path)
    print(f"\n{'='*60}")
    print(f"Processing PDF: {filename}")
    print(f"Total pages: {len(doc)}")
    print(f"{'='*60}\n")

    sections_data = extract_sections_advanced(doc)
    pdf_metadata = doc.metadata
    total_pages = len(doc)
    doc.close()

    if not sections_data:
        raise Exception("No content extracted from PDF")

    upload_id = secrets.token_urlsafe(16)

    # Enrich sections with shared fields needed by rag_service.index_document()
    sections: List[Dict[str, Any]] = []
    for idx, section in enumerate(sections_data):
        sections.append({
            "section_number": idx + 1,
            "section_title": section['section_title'],
            "section_type": section['section_type'],
            "text_content": section['text_content'],      # used by Pinecone only
            "char_count": section['char_count'],
            "start_page": section['start_page'],
            "end_page": section['end_page'],
            "filename": filename,
            "pdf_metadata": {
                "author": pdf_metadata.get("author", ""),
                "title":  pdf_metadata.get("title", ""),
                "subject": pdf_metadata.get("subject", ""),
            },
        })

    # Lightweight metadata record — no text_content stored in MongoDB
    metadata_doc = {
        "user_id": user_id,
        "upload_id": upload_id,
        "filename": filename,
        "total_pages": total_pages,
        "total_sections": len(sections),
        "total_characters": sum(s['char_count'] for s in sections),
        "pdf_metadata": {
            "author": pdf_metadata.get("author", ""),
            "title":  pdf_metadata.get("title", ""),
            "subject": pdf_metadata.get("subject", ""),
        },
        "uploaded_at": datetime.utcnow(),
    }

    return sections, metadata_doc


# ─── Auth Endpoints ───────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message": "PDF Processing API with RAG Search",
        "version": "5.0-PINECONE-ONLY",
        "features": [
            "JWT Auth", "bcrypt passwords", "User Authentication",
            "PDF Processing", "RAG Search", "Pinecone-only text storage"
        ]
    }


@app.post("/register")
async def register_user(request: RegisterRequest):
    try:
        existing_user = users_collection.find_one({"email": request.email})
        if existing_user:
            raise HTTPException(status_code=400, detail="An account with this email already exists.")

        user_id = secrets.token_urlsafe(16)
        password_hash = hash_password(request.password)
        access_token = create_access_token(user_id=user_id, email=request.email)

        user_doc = {
            "user_id": user_id,
            "email": request.email,
            "name": request.name,
            "password_hash": password_hash,
            "created_at": datetime.utcnow(),
            "last_login": datetime.utcnow(),
        }

        users_collection.insert_one(user_doc)

        return {
            "success": True,
            "user_token": access_token,
            "user_id": user_id,
            "name": request.name,
            "email": request.email,
            "message": "Account created successfully. Welcome to Acumen!"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/login")
async def login_user(request: LoginRequest):
    try:
        user = users_collection.find_one({"email": request.email})

        if not user or not verify_password(request.password, user["password_hash"], request.email):
            raise HTTPException(status_code=401, detail="Invalid email or password.")

        access_token = create_access_token(user_id=user["user_id"], email=user["email"])

        users_collection.update_one(
            {"email": request.email},
            {"$set": {"last_login": datetime.utcnow()}}
        )

        return {
            "success": True,
            "user_token": access_token,
            "user_id": user["user_id"],
            "name": user["name"],
            "email": user["email"],
            "message": "Login successful. Welcome back!"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Protected Endpoints ──────────────────────────────────────────────────────

@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    CHANGED:
    1. process_pdf() extracts text in-memory (no MongoDB write for text)
    2. Lightweight metadata saved to pdf_uploads_collection
    3. Sections passed directly to rag_service.index_document() → Pinecone
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        # Step 1: Extract sections in-memory
        sections, metadata_doc = process_pdf(
            tmp_file_path, current_user["user_id"], file.filename
        )
        os.unlink(tmp_file_path)

        # Step 2: Save lightweight metadata to MongoDB (no text_content)
        pdf_uploads_collection.insert_one(metadata_doc)

        # Step 3: Index sections (with text_content) directly into Pinecone
        index_result = rag_service.index_document(
            user_id=current_user["user_id"],
            upload_id=metadata_doc["upload_id"],
            sections=sections,
        )

        return {
            "success": True,
            "upload_id": metadata_doc["upload_id"],
            "sections_indexed": metadata_doc["total_sections"],
            "total_pages": metadata_doc["total_pages"],
            "total_characters": metadata_doc["total_characters"],
            "message": "PDF processed and indexed successfully",
            "rag_indexing": index_result,
        }

    except Exception as e:
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/my-documents")
async def get_my_documents(current_user: dict = Depends(get_current_user)):
    """
    CHANGED: reads from pdf_uploads_collection (lightweight metadata only).
    No text_content is returned — it lives in Pinecone.
    """
    try:
        docs = list(
            pdf_uploads_collection.find(
                {"user_id": current_user["user_id"]},
                {"_id": 0}              # exclude Mongo _id from response
            ).sort("uploaded_at", -1)
        )
        return {
            "success": True,
            "user_id": current_user["user_id"],
            "total_uploads": len(docs),
            "documents": docs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/document/{upload_id}")
async def get_document_details(upload_id: str, current_user: dict = Depends(get_current_user)):
    """
    CHANGED: returns lightweight metadata from pdf_uploads_collection.
    Section text is not stored in MongoDB — use /rag-search to query content.
    """
    try:
        doc = pdf_uploads_collection.find_one(
            {"user_id": current_user["user_id"], "upload_id": upload_id},
            {"_id": 0}
        )
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        return {
            "success": True,
            **doc,
            "note": "Section text is stored in Pinecone. Use /rag-search or /ask to query content."
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/document/{upload_id}")
async def delete_document(upload_id: str, current_user: dict = Depends(get_current_user)):
    """
    CHANGED: deletes from pdf_uploads_collection + Pinecone vectors.
    """
    try:
        result = pdf_uploads_collection.delete_one(
            {"user_id": current_user["user_id"], "upload_id": upload_id}
        )
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Document not found")

        rag_result = rag_service.delete_document_vectors(current_user["user_id"], upload_id)

        return {
            "success": True,
            "upload_id": upload_id,
            "rag_deletion": rag_result,
            "message": "Document deleted successfully"
        }
    except HTTPException:
        raise
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
    CHANGED: text_content now comes directly from Pinecone search results.
    No secondary MongoDB lookup needed.
    """
    try:
        if not query or len(query.strip()) < 3:
            raise HTTPException(status_code=400, detail="Query must be at least 3 characters")

        results = rag_service.search(
            user_id=current_user["user_id"],
            query=query,
            top_k=top_k,
            upload_id=upload_id
        )
        return results

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
async def ask_question(
    query: str,
    top_k: int = 5,
    upload_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    try:
        if not query or len(query.strip()) < 1:
            raise HTTPException(status_code=400, detail="Please enter a message.")

        result = rag_service.generate_answer(
            user_id=current_user["user_id"],
            query=query,
            top_k=top_k,
            upload_id=upload_id
        )
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("message", "Failed to generate answer"))
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rag-stats")
async def get_rag_stats(current_user: dict = Depends(get_current_user)):
    """
    Returns Pinecone index stats + document list from lightweight MongoDB metadata.
    """
    try:
        pinecone_stats = rag_service.get_user_stats(current_user["user_id"])

        docs = list(
            pdf_uploads_collection.find(
                {"user_id": current_user["user_id"]},
                {"_id": 0}
            ).sort("uploaded_at", -1)
        )

        return {
            **pinecone_stats,
            "total_documents": len(docs),
            "documents": docs,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/reset-my-index")
async def reset_my_index(current_user: dict = Depends(get_current_user)):
    """
    Wipes ALL Pinecone vectors AND all pdf_uploads metadata for this user.
    Use this once to clear stale vectors before re-uploading your PDFs.
    """
    try:
        # Delete entire Pinecone index for this user
        pinecone_result = rag_service.delete_user_vectors(current_user["user_id"])

        # Clear all upload metadata from MongoDB
        mongo_result = pdf_uploads_collection.delete_many(
            {"user_id": current_user["user_id"]}
        )

        return {
            "success": True,
            "message": "All vectors and document metadata cleared. You can now re-upload your PDFs.",
            "pinecone": pinecone_result,
            "metadata_records_deleted": mongo_result.deleted_count,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)