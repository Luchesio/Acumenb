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
import bcrypt
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr, field_validator

# Load environment variables
load_dotenv()

# ─── Chunking Configuration ──────────────────────────────────────────────────
# Fix 3: Replace fixed 6-page window with character-based chunking
# Fix 4: Enforce a minimum chunk size so near-empty chunks are discarded
CHUNK_SIZE    = 4000   # target characters per chunk in fallback mode
CHUNK_OVERLAP = 400    # overlap between consecutive chunks (preserves boundary context)
MIN_CHUNK_SIZE = 200   # chunks smaller than this are discarded


def split_text_with_overlap(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    min_chunk_size: int = MIN_CHUNK_SIZE,
) -> list[str]:
    """
    Split *text* into overlapping chunks that respect natural language boundaries.

    Priority for break-points (highest → lowest):
      1. Paragraph break  (\\n\\n)
      2. Sentence end     ('. ')
      3. Line break       (\\n)
      4. Word boundary    (' ')
      5. Hard cut         (arbitrary character position)

    The *chunk_overlap* region is preserved at the start of each new chunk so
    that context spanning a boundary is not lost during retrieval.
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

        # Search for the best break-point within the window
        break_point = -1
        for sep in ['\n\n', '. ', '\n', ' ']:
            bp = text.rfind(sep, start, end)
            if bp != -1 and bp > start:
                break_point = bp + len(sep)   # include separator in the left chunk
                break

        if break_point == -1:
            break_point = end   # hard cut — no natural boundary found

        chunk = text[start:break_point].strip()
        if len(chunk) >= min_chunk_size:
            chunks.append(chunk)

        # Slide forward, stepping back by overlap to capture boundary context
        start = max(start + 1, break_point - chunk_overlap)

        # Snap the new start to the nearest natural boundary within the overlap
        # window so we don't begin mid-sentence when possible.
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
DOCUMENTS_COLLECTION = os.getenv("DOCUMENTS_COLLECTION", "pdf_documents")

# JWT configuration
# IMPORTANT: Set JWT_SECRET_KEY in your .env file — never expose this value.
# Generate a strong key: python -c "import secrets; print(secrets.token_urlsafe(64))"
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not JWT_SECRET_KEY:
    # Fallback for development — in production this MUST be set in .env
    JWT_SECRET_KEY = secrets.token_urlsafe(64)
    print("WARNING: JWT_SECRET_KEY not set in environment. Using ephemeral key — all tokens will be invalidated on restart.")

JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 30  # Token valid for 30 days

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
users_collection.create_index("user_id", unique=True)
documents_collection.create_index("user_id")
documents_collection.create_index([("user_id", 1), ("upload_id", 1)])


# ─── Pydantic Request Models ────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def email_must_be_valid(cls, v: str) -> str:
        v = v.strip().lower()
        # Basic RFC-5322 inspired check — covers the common cases
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


# ─── Password Helpers ───────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    """Hash a password with bcrypt (includes a random salt automatically)."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, stored_hash: str, user_email: str) -> bool:
    """
    Verify a password against a stored hash.

    Migration shim: if the stored hash is an old plain SHA-256 hex digest
    (64 hex chars, no $ prefix) we verify with the old algorithm and
    transparently re-hash to bcrypt so the account is secure going forward.
    """
    # Detect legacy SHA-256 hash (64 lowercase hex chars, no bcrypt $ prefix)
    if len(stored_hash) == 64 and re.fullmatch(r"[0-9a-f]{64}", stored_hash):
        old_hash = hashlib.sha256(plain_password.encode()).hexdigest()
        if old_hash != stored_hash:
            return False
        # Valid — migrate to bcrypt silently
        new_hash = hash_password(plain_password)
        users_collection.update_one(
            {"email": user_email},
            {"$set": {"password_hash": new_hash}}
        )
        return True

    # Standard bcrypt path
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), stored_hash.encode("utf-8"))
    except Exception:
        return False


# ─── JWT Helpers ─────────────────────────────────────────────────────────────

def create_access_token(user_id: str, email: str) -> str:
    """Create a signed JWT that expires in ACCESS_TOKEN_EXPIRE_DAYS days."""
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
    """
    Dependency: decode the JWT from the X-User-Token header,
    then load and return the full user document from MongoDB.

    The X-User-Token header is kept for backward compatibility with the
    rest of the API (upload, search, delete, etc.) — no changes needed there.
    """
    if not user_token:
        raise HTTPException(status_code=401, detail="Authentication required. Please log in.")

    try:
        payload = jwt.decode(user_token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id: Optional[str] = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token payload.")
    except JWTError as exc:
        # Distinguish expired vs. malformed for a better UX message
        error_detail = "Your session has expired. Please log in again."
        if "Signature" in str(exc) or "invalid" in str(exc).lower():
            error_detail = "Invalid token. Please log in again."
        raise HTTPException(status_code=401, detail=error_detail)

    user = users_collection.find_one({"user_id": user_id})
    if not user:
        raise HTTPException(status_code=401, detail="User account not found.")

    return user


# ─── PDF Processing (unchanged) ──────────────────────────────────────────────

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


def extract_sections_advanced(doc):
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
        # ── Fix 1 & 4: Structure-based path ─────────────────────────────────
        # Sub-chunk any section that is too large to embed faithfully, and
        # discard any section whose content falls below MIN_CHUNK_SIZE.
        for i, match in enumerate(detected_sections):
            section_title = match.group(0).strip()
            start_pos = match.start()
            end_pos = (
                detected_sections[i + 1].start()
                if i + 1 < len(detected_sections)
                else len(full_text)
            )
            section_text = full_text[start_pos:end_pos].strip()

            # Fix 4: skip near-empty sections
            if len(section_text) < MIN_CHUNK_SIZE:
                print(f"  Skipping tiny section '{section_title}' ({len(section_text)} chars)")
                continue

            chars_before = len(full_text[:start_pos])
            start_page = max(1, int(chars_before / chars_per_page) + 1)
            pages_in_section = max(1, int(len(section_text) / chars_per_page))
            end_page = min(len(doc), start_page + pages_in_section)

            # Fix 1: sub-chunk large sections so no single embedding is lossy
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
        # ── Fix 2, 3 & 4: Fallback path ─────────────────────────────────────
        # Replace the fixed 6-page window with character-based chunking that
        # uses sentence/paragraph-aware overlap so boundary context is never lost.
        print("No clear sections detected. Using overlap-aware character chunking...")

        if len(doc) <= 5:
            # Tiny document: one chunk is fine, but still enforce minimum size
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
            # Fix 3: character-based overlap chunking instead of 6-page windows
            text_chunks = split_text_with_overlap(full_text)
            cumulative = 0
            for chunk_text in text_chunks:
                # Estimate page range from character position
                start_page = max(1, int(cumulative / chars_per_page) + 1)
                end_page = min(
                    len(doc),
                    int((cumulative + len(chunk_text)) / chars_per_page) + 1
                )

                # Infer a title from the first meaningful line
                chunk_title = None
                for line in chunk_text.split('\n')[:10]:
                    line = line.strip()
                    if 10 <= len(line) <= 100 and not line.endswith('.'):
                        chunk_title = line
                        break
                if not chunk_title:
                    chunk_title = f"Pages {start_page}–{end_page}"

                # Fix 4: skip if still below minimum (edge-case guard)
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

                # Advance past the non-overlapping portion
                cumulative += max(1, len(chunk_text) - CHUNK_OVERLAP)

    total_extracted = sum(s['char_count'] for s in sections_data)
    print(f"\nExtraction Summary:")
    print(f"- Total sections: {len(sections_data)}")
    print(f"- Original text: {len(full_text)} chars")
    print(f"- Extracted text: {total_extracted} chars")
    print(f"- Coverage: {(total_extracted/len(full_text)*100):.1f}%")

    return sections_data


def process_pdf_to_mongodb(pdf_path: str, user_id: str, filename: str):
    try:
        doc = fitz.open(pdf_path)
        print(f"\n{'='*60}")
        print(f"Processing PDF: {filename}")
        print(f"Total pages: {len(doc)}")
        print(f"{'='*60}\n")

        sections_data = extract_sections_advanced(doc)
        metadata = doc.metadata
        total_pages = len(doc)
        doc.close()

        if not sections_data:
            raise Exception("No content extracted from PDF")

        upload_id = secrets.token_urlsafe(16)
        documents = []

        for idx, section in enumerate(sections_data):
            documents.append({
                "user_id": user_id,
                "upload_id": upload_id,
                "filename": filename,
                "section_number": idx + 1,
                "section_title": section['section_title'],
                "section_type": section['section_type'],
                "text_content": section['text_content'],
                "char_count": section['char_count'],
                "start_page": section['start_page'],
                "end_page": section['end_page'],
                "total_pages": total_pages,
                "pdf_metadata": {
                    "author": metadata.get("author", ""),
                    "title": metadata.get("title", ""),
                    "subject": metadata.get("subject", ""),
                },
                "uploaded_at": datetime.utcnow()
            })

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


# ─── Auth Endpoints ──────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message": "PDF Processing API with RAG Search",
        "version": "4.0-SECURE",
        "features": ["JWT Auth", "bcrypt passwords", "User Authentication", "PDF Processing", "RAG Search"]
    }


@app.post("/register")
async def register_user(request: RegisterRequest):
    """
    Register a new user.

    Accepts a JSON body: { "name": "...", "email": "...", "password": "..." }
    Returns a JWT access token on success.
    """
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
    """
    Log in an existing user.

    Accepts a JSON body: { "email": "...", "password": "..." }
    Returns a JWT access token on success.
    """
    try:
        user = users_collection.find_one({"email": request.email})

        # Use a constant-time-safe check so we don't leak whether the email exists
        if not user or not verify_password(request.password, user["password_hash"], request.email):
            raise HTTPException(status_code=401, detail="Invalid email or password.")

        # Issue a fresh token on every login
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


# ─── Protected Endpoints (unchanged logic, now secured via JWT) ──────────────

@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        result = process_pdf_to_mongodb(tmp_file_path, current_user["user_id"], file.filename)
        os.unlink(tmp_file_path)

        if result["success"]:
            index_result = rag_service.index_document(current_user["user_id"], result["upload_id"])
            result["rag_indexing"] = index_result

        return result

    except Exception as e:
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/my-documents")
async def get_my_documents(current_user: dict = Depends(get_current_user)):
    try:
        pipeline = [
            {"$match": {"user_id": current_user["user_id"]}},
            {"$group": {
                "_id": "$upload_id",
                "filename": {"$first": "$filename"},
                "uploaded_at": {"$first": "$uploaded_at"},
                "total_sections": {"$sum": 1},
                "total_pages": {"$first": "$total_pages"},
                "total_characters": {"$sum": "$char_count"},
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
async def get_document_details(upload_id: str, current_user: dict = Depends(get_current_user)):
    try:
        sections = list(documents_collection.find(
            {"user_id": current_user["user_id"], "upload_id": upload_id}
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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/document/{upload_id}")
async def delete_document(upload_id: str, current_user: dict = Depends(get_current_user)):
    try:
        result = documents_collection.delete_many(
            {"user_id": current_user["user_id"], "upload_id": upload_id}
        )
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Document not found")
        rag_result = rag_service.delete_document_vectors(current_user["user_id"], upload_id)
        return {
            "success": True,
            "deleted_sections": result.deleted_count,
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
    try:
        if not query or len(query.strip()) < 3:
            raise HTTPException(status_code=400, detail="Query must be at least 3 characters")
        results = rag_service.search(
            user_id=current_user["user_id"], query=query, top_k=top_k, upload_id=upload_id
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
            user_id=current_user["user_id"], query=query, top_k=top_k, upload_id=upload_id
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
    try:
        return rag_service.get_user_stats(current_user["user_id"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reindex-document/{upload_id}")
async def reindex_document(upload_id: str, current_user: dict = Depends(get_current_user)):
    try:
        doc_exists = documents_collection.find_one(
            {"user_id": current_user["user_id"], "upload_id": upload_id}
        )
        if not doc_exists:
            raise HTTPException(status_code=404, detail="Document not found")
        return rag_service.index_document(current_user["user_id"], upload_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reindex-all")
async def reindex_all_documents(current_user: dict = Depends(get_current_user)):
    try:
        pipeline = [
            {"$match": {"user_id": current_user["user_id"]}},
            {"$group": {"_id": "$upload_id"}}
        ]
        upload_ids = [doc["_id"] for doc in documents_collection.aggregate(pipeline)]
        if not upload_ids:
            return {"success": True, "message": "No documents to index", "indexed_documents": 0}

        indexed_count = 0
        errors = []
        for uid in upload_ids:
            result = rag_service.index_document(current_user["user_id"], uid)
            if result["success"]:
                indexed_count += 1
            else:
                errors.append({"upload_id": uid, "error": result.get("message", "Unknown error")})

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