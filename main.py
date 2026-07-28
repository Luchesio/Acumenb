import fitz
import json
import re
import os
import unicodedata
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends, Form
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import Optional, List
import tempfile
import time
import secrets
import hashlib
import ssl
from rag_service import get_rag_service
from voice_service import get_voice_service
import bcrypt
from jose import JWTError, jwt
from pydantic import BaseModel, field_validator
from bson import ObjectId
import cloudinary
import cloudinary.uploader
import cloudinary.utils
from langchain_text_splitters import RecursiveCharacterTextSplitter

from google.oauth2 import id_token as google_id_token
from google.auth.transport import requests as google_requests

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── MongoDB configuration ────────────────────────────────────────────────────
MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME")
USERS_COLLECTION = os.getenv("USERS_COLLECTION", "users")
DOCUMENTS_COLLECTION = os.getenv("DOCUMENTS_COLLECTION", "pdf_documents")

# ─── JWT configuration ────────────────────────────────────────────────────────
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not JWT_SECRET_KEY:
    JWT_SECRET_KEY = secrets.token_urlsafe(64)
    print("WARNING: JWT_SECRET_KEY not set. Using ephemeral key — tokens invalidated on restart.")

JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 30

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
if not GOOGLE_CLIENT_ID:
    print("WARNING: GOOGLE_CLIENT_ID not set. Google Sign-In will be disabled.")

# ─── Upload limits ────────────────────────────────────────────────────────────
MB = 1024 * 1024
MAX_FILES_PER_UPLOAD = int(os.getenv("MAX_FILES_PER_UPLOAD", "10"))
MAX_FILE_SIZE_MB = float(os.getenv("MAX_FILE_SIZE_MB", "25"))
MAX_BATCH_SIZE_MB = float(os.getenv("MAX_BATCH_SIZE_MB", "60"))
MAX_USER_STORAGE_MB = float(os.getenv("MAX_USER_STORAGE_MB", "200"))

# ─── Chunking configuration ───────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
ENABLE_OCR = os.getenv("ENABLE_OCR", "true").lower() == "true"
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "6"))
MAX_BULK_OPEN = int(os.getenv("MAX_BULK_OPEN", "10"))

# Vercel rejects request bodies over 4.5 MB before they reach this app, so the
# ceiling here is deliberately below that to leave room for the rest of the form.
MAX_ATTACHMENTS = int(os.getenv("MAX_ATTACHMENTS", "4"))
MAX_ATTACHMENT_MB = float(os.getenv("MAX_ATTACHMENT_MB", "3.5"))
ALLOWED_ATTACHMENT_TYPES = {
    "image/jpeg", "image/png", "image/webp", "image/heic", "image/heif",
    "application/pdf",
}

PDF_URL_STRATEGY = os.getenv("PDF_URL_STRATEGY", "download").strip().lower()
PDF_URL_TTL_MINUTES = int(os.getenv("PDF_URL_TTL_MINUTES", "10"))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    keep_separator=True,
    separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
)

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True,
)

context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
context.minimum_version = ssl.TLSVersion.TLSv1_2

client = MongoClient(MONGO_URI, tls=True, tlsAllowInvalidCertificates=True)
db = client[DATABASE_NAME]
users_collection = db[USERS_COLLECTION]
documents_collection = db[DOCUMENTS_COLLECTION]

rag_service = get_rag_service()
voice_service = get_voice_service()

users_collection.create_index("email", unique=True)
users_collection.create_index("user_id", unique=True)
documents_collection.create_index("user_id")
documents_collection.create_index([("user_id", 1), ("upload_id", 1)])


# ─── Pydantic Models ──────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def email_must_be_valid(cls, v: str) -> str:
        v = v.strip().lower()
        if not re.match(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$", v):
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


class GoogleAuthRequest(BaseModel):
    credential: str


class ChatTurn(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def normalise_role(cls, v: str) -> str:
        return "user" if (v or "").strip().lower() == "user" else "assistant"

    @field_validator("content")
    @classmethod
    def trim_content(cls, v: str) -> str:
        return (v or "").strip()[:4000]


class AskRequest(BaseModel):
    history: List[ChatTurn] = []


# ─── Password helpers ─────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, stored_hash: str, user_email: str) -> bool:
    if len(stored_hash) == 64 and re.fullmatch(r"[0-9a-f]{64}", stored_hash):
        if hashlib.sha256(plain_password.encode()).hexdigest() != stored_hash:
            return False
        users_collection.update_one(
            {"email": user_email},
            {"$set": {"password_hash": hash_password(plain_password)}}
        )
        return True
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), stored_hash.encode("utf-8"))
    except Exception:
        return False


# ─── JWT helpers ──────────────────────────────────────────────────────────────

def create_access_token(user_id: str, email: str) -> str:
    expire = datetime.utcnow() + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    return jwt.encode(
        {"sub": user_id, "email": email, "exp": expire, "iat": datetime.utcnow()},
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM,
    )


async def get_current_user(
    user_token: Optional[str] = Header(None, alias="X-User-Token")
):
    if not user_token:
        raise HTTPException(status_code=401, detail="Authentication required. Please log in.")
    try:
        payload = jwt.decode(user_token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token payload.")
    except JWTError as exc:
        detail = "Your session has expired. Please log in again."
        if "Signature" in str(exc) or "invalid" in str(exc).lower():
            detail = "Invalid token. Please log in again."
        raise HTTPException(status_code=401, detail=detail)
    user = users_collection.find_one({"user_id": user_id})
    if not user:
        raise HTTPException(status_code=401, detail="User account not found.")
    return user


# ─── Cloudinary helpers ───────────────────────────────────────────────────────

def upload_chunks_to_cloudinary(chunks: list, user_id: str, upload_id: str, filename: str) -> dict:
    payload = json.dumps({
        "upload_id": upload_id,
        "user_id": user_id,
        "filename": filename,
        "created_at": datetime.utcnow().isoformat(),
        "chunks": chunks,
    }, ensure_ascii=False).encode("utf-8")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp.write(payload)
        tmp_path = tmp.name

    try:
        result = cloudinary.uploader.upload(
            tmp_path,
            resource_type="raw",
            folder=f"acumen/{user_id}/text",
            public_id=f"{upload_id}_chunks",
            overwrite=True,
        )
    finally:
        os.unlink(tmp_path)

    return {
        "url": result["secure_url"],
        "public_id": result["public_id"],
        "bytes": result.get("bytes", 0),
    }


def upload_pdf_to_cloudinary(pdf_path: str, user_id: str, upload_id: str) -> dict:
    result = cloudinary.uploader.upload(
        pdf_path,
        resource_type="raw",
        type="authenticated",
        folder=f"acumen/{user_id}/pdf",
        public_id=f"{upload_id}.pdf",
        overwrite=True,
    )
    return {
        "url": result["secure_url"],
        "public_id": result["public_id"],
        "bytes": result.get("bytes", 0),
    }


def signed_pdf_url(public_id: str) -> str:
    """Two ways to hand out a private file. Both read assets uploaded as
    type=authenticated, so switching strategies needs no re-upload.

    download — Cloudinary's download API with an expiring signature. Works on
               every plan. Default.
    signed   — a signed delivery URL. Renders inline, but some accounts refuse
               signed delivery of raw assets with a 401.
    """
    if PDF_URL_STRATEGY == "signed":
        url, _ = cloudinary.utils.cloudinary_url(
            public_id,
            resource_type="raw",
            type="authenticated",
            sign_url=True,
            secure=True,
        )
        return url

    return cloudinary.utils.private_download_url(
        public_id,
        None,
        resource_type="raw",
        type="authenticated",
        expires_at=int(time.time()) + PDF_URL_TTL_MINUTES * 60,
    )


def delete_cloudinary_resources(public_ids: list) -> None:
    for pid in public_ids:
        for delivery_type in ("upload", "authenticated"):
            try:
                res = cloudinary.uploader.destroy(pid, resource_type="raw", type=delivery_type)
                if res.get("result") == "ok":
                    break
            except Exception as e:
                print(f"Cloudinary delete failed for {pid} ({delivery_type}): {e}")


def fetch_chunks_from_cloudinary(text_url: str) -> list:
    try:
        resp = requests.get(text_url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data.get("chunks") or data.get("sections", [])
    except Exception as e:
        print(f"Failed to fetch chunks from Cloudinary ({text_url}): {e}")
        return []


# ─── PDF text extraction ──────────────────────────────────────────────────────

LIGATURES = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl",
    "\ufb05": "st", "\ufb06": "st", "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-", "\u00a0": " ",
}


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", "")
    text = unicodedata.normalize("NFKC", text)
    for src, dst in LIGATURES.items():
        text = text.replace(src, dst)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_page_text(page) -> str:
    text = ""
    try:
        text = page.get_text("text", sort=True) or ""
    except Exception:
        text = page.get_text("text") or ""

    if len(text.strip()) < 20:
        try:
            blocks = page.get_text("blocks", sort=True)
            text = "\n".join(b[4] for b in blocks if len(b) >= 5 and isinstance(b[4], str))
        except Exception:
            pass

    if len(text.strip()) < 20:
        try:
            data = page.get_text("dict")
            lines = []
            for block in data.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    joined = "".join(s.get("text", "") for s in line.get("spans", []))
                    if joined.strip():
                        lines.append(joined)
            text = "\n".join(lines)
        except Exception:
            pass

    if len(text.strip()) < 20 and ENABLE_OCR:
        try:
            tp = page.get_textpage_ocr(dpi=200, full=True)
            text = page.get_text("text", textpage=tp) or text
        except Exception:
            pass

    return normalize_text(text)


def extract_pages(pdf_path: str) -> tuple:
    pages = []
    doc = fitz.open(pdf_path)
    try:
        metadata = doc.metadata or {}
        total_pages = len(doc)
        for i, page in enumerate(doc):
            pages.append({"page_number": i + 1, "text": extract_page_text(page)})
    finally:
        doc.close()
    return pages, metadata, total_pages


def build_chunk_title(text: str, start_page: int, end_page: int) -> str:
    page_ref = f"Page {start_page}" if start_page == end_page else f"Pages {start_page}-{end_page}"
    first_line = next(
        (l.strip() for l in text.split("\n")[:3] if 8 <= len(l.strip()) <= 90),
        ""
    )
    return f"{page_ref} — {first_line}" if first_line else page_ref


def chunk_pages(pages: list) -> list:
    full_text = ""
    spans = []
    for page in pages:
        if not page["text"]:
            continue
        start = len(full_text)
        full_text += page["text"] + "\n\n"
        spans.append((start, len(full_text), page["page_number"]))

    if not full_text.strip():
        return []

    raw_chunks = text_splitter.split_text(full_text)
    chunks = []
    cursor = 0

    for index, raw in enumerate(raw_chunks):
        cleaned = raw.strip()
        if not cleaned:
            continue

        probe = raw[:60]
        position = full_text.find(probe, cursor)
        if position == -1:
            position = full_text.find(probe)
        if position == -1:
            position = cursor
        end = position + len(raw)
        cursor = position + max(1, len(raw) - CHUNK_OVERLAP)

        matched = [pg for s, e, pg in spans if position < e and end > s]
        start_page = min(matched) if matched else 1
        end_page = max(matched) if matched else 1

        chunks.append({
            "chunk_index": len(chunks) + 1,
            "text": cleaned,
            "chunk_title": build_chunk_title(cleaned, start_page, end_page),
            "start_page": start_page,
            "end_page": end_page,
            "char_count": len(cleaned),
        })

    return chunks


def process_pdf(pdf_path: str, user_id: str, filename: str, upload_id: str, file_size: int) -> dict:
    pages, metadata, total_pages = extract_pages(pdf_path)
    chunks = chunk_pages(pages)

    if not chunks:
        raise Exception(
            "No readable text found. The PDF appears to be a scan or image-only file. "
            "Enable OCR on the server or upload a text-based PDF."
        )

    text_cl = upload_chunks_to_cloudinary(chunks, user_id, upload_id, filename)

    pdf_cl = {"url": "", "public_id": "", "bytes": 0}
    try:
        pdf_cl = upload_pdf_to_cloudinary(pdf_path, user_id, upload_id)
    except Exception as e:
        print(f"Original PDF could not be stored for {upload_id}: {e}")

    record = {
        "user_id": user_id,
        "upload_id": upload_id,
        "filename": filename,
        "file_size_bytes": file_size,
        "total_pages": total_pages,
        "total_chunks": len(chunks),
        "total_characters": sum(c["char_count"] for c in chunks),
        "pages_without_text": sum(1 for p in pages if not p["text"]),
        "pdf_metadata": {
            "author": metadata.get("author", ""),
            "title": metadata.get("title", ""),
            "subject": metadata.get("subject", ""),
        },
        "text_cloudinary_url": text_cl["url"],
        "text_cloudinary_public_id": text_cl["public_id"],
        "text_bytes": text_cl.get("bytes", 0),
        "pdf_cloudinary_public_id": pdf_cl["public_id"],
        "pdf_bytes": pdf_cl.get("bytes", 0),
        "has_pdf": bool(pdf_cl["public_id"]),
        "uploaded_at": datetime.utcnow(),
    }

    inserted = documents_collection.insert_one(record)

    return {
        "success": True,
        "upload_id": upload_id,
        "filename": filename,
        "document_ref": str(inserted.inserted_id),
        "total_chunks": len(chunks),
        "sections_inserted": len(chunks),
        "total_pages": total_pages,
        "total_characters": record["total_characters"],
        "pages_without_text": record["pages_without_text"],
        "text_cloudinary_url": text_cl["url"],
        "has_pdf": bool(pdf_cl["public_id"]),
        "message": "PDF chunked and stored successfully",
    }


# ─── Storage quota helpers ────────────────────────────────────────────────────

def get_user_storage_bytes(user_id: str) -> int:
    result = list(documents_collection.aggregate([
        {"$match": {"user_id": user_id}},
        {"$group": {
            "_id": "$upload_id",
            "size": {"$first": "$file_size_bytes"},
            "text": {"$first": "$text_bytes"},
        }},
        {"$group": {"_id": None, "total": {"$sum": {"$add": [
            {"$ifNull": ["$size", 0]},
            {"$ifNull": ["$text", 0]},
        ]}}}},
    ]))
    return int(result[0]["total"]) if result else 0


def limit_error(code: str, message: str, **extra) -> HTTPException:
    return HTTPException(status_code=413, detail={"code": code, "message": message, **extra})


# ─── Auth endpoints ───────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message": "ACUMEN PDF API",
        "version": "6.0",
        "storage": "MongoDB (metadata) + Cloudinary (chunks) + Pinecone (vectors)",
        "chunking": f"RecursiveCharacterTextSplitter({CHUNK_SIZE}/{CHUNK_OVERLAP})",
    }


@app.post("/register")
async def register_user(request: RegisterRequest):
    try:
        if users_collection.find_one({"email": request.email}):
            raise HTTPException(status_code=400, detail="An account with this email already exists.")
        user_id = secrets.token_urlsafe(16)
        token = create_access_token(user_id=user_id, email=request.email)
        users_collection.insert_one({
            "user_id": user_id, "email": request.email, "name": request.name,
            "password_hash": hash_password(request.password),
            "auth_provider": "email",
            "created_at": datetime.utcnow(), "last_login": datetime.utcnow(),
        })
        return {"success": True, "user_token": token, "user_id": user_id,
                "name": request.name, "email": request.email,
                "message": "Account created successfully. Welcome to Acumen!"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/login")
async def login_user(request: LoginRequest):
    try:
        user = users_collection.find_one({"email": request.email})
        if not user:
            raise HTTPException(status_code=401, detail="Invalid email or password.")

        if user.get("auth_provider") == "google" and not user.get("password_hash"):
            raise HTTPException(
                status_code=400,
                detail="This account uses Google Sign-In. Please use the 'Continue with Google' button."
            )

        if not verify_password(request.password, user["password_hash"], request.email):
            raise HTTPException(status_code=401, detail="Invalid email or password.")

        token = create_access_token(user_id=user["user_id"], email=user["email"])
        users_collection.update_one({"email": request.email}, {"$set": {"last_login": datetime.utcnow()}})
        return {"success": True, "user_token": token, "user_id": user["user_id"],
                "name": user["name"], "email": user["email"], "message": "Login successful."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/auth/google")
async def google_auth(request: GoogleAuthRequest):
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=503, detail="Google Sign-In is not configured on this server.")

    try:
        idinfo = google_id_token.verify_oauth2_token(
            request.credential,
            google_requests.Request(),
            GOOGLE_CLIENT_ID,
        )
    except ValueError as e:
        raise HTTPException(status_code=401, detail=f"Invalid Google token: {str(e)}")

    email = idinfo.get("email", "").strip().lower()
    google_sub = idinfo.get("sub")
    google_name = idinfo.get("name") or email.split("@")[0]
    email_verified = idinfo.get("email_verified", False)

    if not email or not google_sub:
        raise HTTPException(status_code=401, detail="Google token is missing required fields.")

    if not email_verified:
        raise HTTPException(status_code=400, detail="Google account email is not verified.")

    try:
        existing_user = users_collection.find_one({"email": email})

        if existing_user:
            update_fields: dict = {"last_login": datetime.utcnow()}
            if not existing_user.get("google_id"):
                update_fields["google_id"] = google_sub
            users_collection.update_one({"email": email}, {"$set": update_fields})

            user_id = existing_user["user_id"]
            display_name = existing_user["name"]
            message = "Signed in with Google successfully."
        else:
            user_id = secrets.token_urlsafe(16)
            display_name = google_name
            users_collection.insert_one({
                "user_id": user_id,
                "email": email,
                "name": display_name,
                "google_id": google_sub,
                "password_hash": None,
                "auth_provider": "google",
                "created_at": datetime.utcnow(),
                "last_login": datetime.utcnow(),
            })
            message = "Account created with Google. Welcome to Acumen!"

        token = create_access_token(user_id=user_id, email=email)
        return {
            "success": True,
            "user_token": token,
            "user_id": user_id,
            "name": display_name,
            "email": email,
            "message": message,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Upload endpoints ─────────────────────────────────────────────────────────

@app.get("/upload-limits")
async def get_upload_limits(current_user: dict = Depends(get_current_user)):
    used = get_user_storage_bytes(current_user["user_id"])
    return {
        "success": True,
        "max_files_per_upload": MAX_FILES_PER_UPLOAD,
        "max_file_size_mb": MAX_FILE_SIZE_MB,
        "max_batch_size_mb": MAX_BATCH_SIZE_MB,
        "max_storage_mb": MAX_USER_STORAGE_MB,
        "storage_used_bytes": used,
        "storage_used_mb": round(used / MB, 2),
        "storage_remaining_mb": round(max(0.0, MAX_USER_STORAGE_MB - used / MB), 2),
    }


@app.post("/upload-pdfs")
async def upload_pdfs(
    files: List[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user)
):
    if not files:
        raise HTTPException(status_code=400, detail="No files received.")

    if len(files) > MAX_FILES_PER_UPLOAD:
        raise limit_error(
            "TOO_MANY_FILES",
            f"You can upload up to {MAX_FILES_PER_UPLOAD} PDFs at a time. You selected {len(files)}.",
            limit=MAX_FILES_PER_UPLOAD,
            attempted=len(files),
        )

    invalid = [f.filename for f in files if not (f.filename or "").lower().endswith(".pdf")]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail={"code": "INVALID_TYPE", "message": "Only PDF files are allowed.", "files": invalid},
        )

    payloads = []
    oversized = []
    total_bytes = 0

    for f in files:
        data = await f.read()
        size = len(data)
        total_bytes += size
        if size > MAX_FILE_SIZE_MB * MB:
            oversized.append({"filename": f.filename, "size_mb": round(size / MB, 2)})
        else:
            payloads.append({"filename": f.filename, "data": data, "size": size})

    if oversized:
        raise limit_error(
            "FILE_TOO_LARGE",
            f"Each PDF must be {MAX_FILE_SIZE_MB:g} MB or smaller.",
            limit_mb=MAX_FILE_SIZE_MB,
            files=oversized,
        )

    if total_bytes > MAX_BATCH_SIZE_MB * MB:
        raise limit_error(
            "BATCH_TOO_LARGE",
            f"This batch is {round(total_bytes / MB, 2)} MB. The limit is {MAX_BATCH_SIZE_MB:g} MB per upload.",
            limit_mb=MAX_BATCH_SIZE_MB,
            attempted_mb=round(total_bytes / MB, 2),
        )

    used_bytes = get_user_storage_bytes(current_user["user_id"])
    if used_bytes + total_bytes > MAX_USER_STORAGE_MB * MB:
        raise limit_error(
            "STORAGE_LIMIT_REACHED",
            f"This upload would exceed your {MAX_USER_STORAGE_MB:g} MB storage limit. "
            f"Delete some documents to free up space.",
            limit_mb=MAX_USER_STORAGE_MB,
            used_mb=round(used_bytes / MB, 2),
            remaining_mb=round(max(0.0, MAX_USER_STORAGE_MB - used_bytes / MB), 2),
            attempted_mb=round(total_bytes / MB, 2),
        )

    results = []
    for item in payloads:
        upload_id = secrets.token_urlsafe(16)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(item["data"])
                tmp_path = tmp.name

            result = await run_in_threadpool(
                process_pdf,
                tmp_path,
                current_user["user_id"],
                item["filename"],
                upload_id,
                item["size"],
            )

            index_result = await run_in_threadpool(
                rag_service.index_document,
                current_user["user_id"],
                upload_id,
                result["text_cloudinary_url"],
            )
            result["rag_indexed"] = index_result.get("success", False)
            result["rag_indexing"] = index_result
            if not result["rag_indexed"]:
                result["rag_indexing_warning"] = (
                    "Document saved but AI indexing failed. Use Re-index to retry."
                )
            results.append(result)

        except Exception as e:
            documents_collection.delete_many(
                {"user_id": current_user["user_id"], "upload_id": upload_id}
            )
            results.append({
                "success": False,
                "filename": item["filename"],
                "upload_id": upload_id,
                "message": str(e),
            })
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    successful = [r for r in results if r.get("success")]
    used_after = get_user_storage_bytes(current_user["user_id"])

    return {
        "success": len(successful) > 0,
        "total_files": len(payloads),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "total_chunks": sum(r.get("total_chunks", 0) for r in successful),
        "storage_used_mb": round(used_after / MB, 2),
        "storage_remaining_mb": round(max(0.0, MAX_USER_STORAGE_MB - used_after / MB), 2),
        "results": results,
        "message": f"{len(successful)} of {len(payloads)} PDFs processed",
    }


@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    return await upload_pdfs(files=[file], current_user=current_user)


# ─── Document endpoints ───────────────────────────────────────────────────────

@app.get("/my-documents")
async def get_my_documents(current_user: dict = Depends(get_current_user)):
    try:
        pipeline = [
            {"$match": {"user_id": current_user["user_id"]}},
            {"$group": {
                "_id": "$upload_id",
                "filename": {"$first": "$filename"},
                "uploaded_at": {"$first": "$uploaded_at"},
                "record_count": {"$sum": 1},
                "total_chunks": {"$first": "$total_chunks"},
                "total_pages": {"$first": "$total_pages"},
                "total_characters": {"$sum": {"$ifNull": ["$total_characters", "$char_count"]}},
                "file_size_bytes": {"$first": "$file_size_bytes"},
                "pdf_title": {"$first": "$pdf_metadata.title"},
                "text_cloudinary_url": {"$first": "$text_cloudinary_url"},
                "has_pdf": {"$first": "$has_pdf"},
            }},
            {"$sort": {"uploaded_at": -1}},
        ]
        docs = list(documents_collection.aggregate(pipeline))
        for d in docs:
            d["total_sections"] = d.get("total_chunks") or d.get("record_count", 0)
            d["size_mb"] = round((d.get("file_size_bytes") or 0) / MB, 2)

        used = get_user_storage_bytes(current_user["user_id"])
        return {
            "success": True,
            "user_id": current_user["user_id"],
            "total_uploads": len(docs),
            "storage_used_mb": round(used / MB, 2),
            "storage_remaining_mb": round(max(0.0, MAX_USER_STORAGE_MB - used / MB), 2),
            "max_storage_mb": MAX_USER_STORAGE_MB,
            "documents": docs,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/document/{upload_id}")
async def get_document_details(upload_id: str, current_user: dict = Depends(get_current_user)):
    try:
        doc = documents_collection.find_one(
            {"user_id": current_user["user_id"], "upload_id": upload_id}
        )
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        doc["_id"] = str(doc["_id"])
        chunks = fetch_chunks_from_cloudinary(doc.get("text_cloudinary_url", ""))
        return {
            "success": True,
            "upload_id": upload_id,
            "filename": doc["filename"],
            "total_chunks": doc.get("total_chunks", len(chunks)),
            "total_pages": doc.get("total_pages"),
            "total_characters": doc.get("total_characters", 0),
            "text_cloudinary_url": doc.get("text_cloudinary_url"),
            "document": doc,
            "chunks": chunks,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ViewLinksRequest(BaseModel):
    upload_ids: List[str]


@app.post("/documents/view-links")
async def documents_view_links(
    request: ViewLinksRequest,
    current_user: dict = Depends(get_current_user)
):
    ids = list(dict.fromkeys(request.upload_ids))[:MAX_BULK_OPEN]
    if not ids:
        raise HTTPException(status_code=400, detail="No documents specified.")

    docs = list(documents_collection.aggregate([
        {"$match": {"user_id": current_user["user_id"], "upload_id": {"$in": ids}}},
        {"$group": {
            "_id": "$upload_id",
            "filename": {"$first": "$filename"},
            "pdf_cloudinary_public_id": {"$first": "$pdf_cloudinary_public_id"},
        }},
    ]))

    links, unavailable = [], []
    found = {d["_id"] for d in docs}

    for doc in docs:
        public_id = doc.get("pdf_cloudinary_public_id")
        if not public_id:
            unavailable.append({"upload_id": doc["_id"], "filename": doc.get("filename", ""),
                                "reason": "Original file was not stored"})
            continue
        try:
            links.append({"upload_id": doc["_id"], "filename": doc.get("filename", ""),
                          "url": signed_pdf_url(public_id)})
        except Exception as e:
            unavailable.append({"upload_id": doc["_id"], "filename": doc.get("filename", ""),
                                "reason": f"Could not sign link: {e}"})

    for missing in [i for i in ids if i not in found]:
        unavailable.append({"upload_id": missing, "filename": "", "reason": "Document not found"})

    order = {uid: i for i, uid in enumerate(ids)}
    links.sort(key=lambda l: order.get(l["upload_id"], 0))

    return {"success": True, "links": links, "unavailable": unavailable}


@app.get("/document/{upload_id}/view")
async def view_document(upload_id: str, current_user: dict = Depends(get_current_user)):
    doc = documents_collection.find_one(
        {"user_id": current_user["user_id"], "upload_id": upload_id},
        {"pdf_cloudinary_public_id": 1, "filename": 1}
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    public_id = doc.get("pdf_cloudinary_public_id")
    if not public_id:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "NO_ORIGINAL_PDF",
                "message": (
                    "The original file for this document was not kept. Documents uploaded "
                    "before this feature only stored their extracted text. Re-upload it to "
                    "enable viewing."
                ),
            },
        )

    try:
        return {
            "success": True,
            "filename": doc.get("filename", ""),
            "url": signed_pdf_url(public_id),
            "strategy": PDF_URL_STRATEGY,
            "expires_in_seconds": None if PDF_URL_STRATEGY == "signed" else PDF_URL_TTL_MINUTES * 60,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not generate a view link: {str(e)}")


@app.post("/cleanup-orphans")
async def cleanup_orphans(current_user: dict = Depends(get_current_user)):
    try:
        return rag_service.cleanup_orphan_vectors(current_user["user_id"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/document/{upload_id}")
async def delete_document(upload_id: str, current_user: dict = Depends(get_current_user)):
    try:
        sample = documents_collection.find_one(
            {"user_id": current_user["user_id"], "upload_id": upload_id},
            {"text_cloudinary_public_id": 1, "pdf_cloudinary_public_id": 1}
        )
        if not sample:
            raise HTTPException(status_code=404, detail="Document not found")

        rag_result = rag_service.delete_document_vectors(current_user["user_id"], upload_id)
        if not rag_result.get("success"):
            raise HTTPException(
                status_code=502,
                detail=(
                    "Could not remove this document from the search index, so it was not "
                    "deleted. Nothing was lost — please try again. "
                    f"({rag_result.get('message', 'unknown error')})"
                ),
            )

        delete_cloudinary_resources([
            pid for pid in [
                sample.get("text_cloudinary_public_id"),
                sample.get("pdf_cloudinary_public_id"),
            ] if pid
        ])

        db_result = documents_collection.delete_many(
            {"user_id": current_user["user_id"], "upload_id": upload_id}
        )
        used = get_user_storage_bytes(current_user["user_id"])

        return {
            "success": True,
            "deleted_records": db_result.deleted_count,
            "rag_deletion": rag_result,
            "storage_used_mb": round(used / MB, 2),
            "storage_remaining_mb": round(max(0.0, MAX_USER_STORAGE_MB - used / MB), 2),
            "message": "Document deleted from all storage layers",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── RAG endpoints ────────────────────────────────────────────────────────────

@app.get("/rag-search")
async def rag_search(
    query: str,
    top_k: int = 12,
    upload_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    try:
        if not query or len(query.strip()) < 3:
            raise HTTPException(status_code=400, detail="Query must be at least 3 characters")
        return rag_service.search(
            user_id=current_user["user_id"], query=query, top_k=top_k, upload_id=upload_id
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask-with-files")
async def ask_with_files(
    query: str = "",
    top_k: int = 12,
    upload_id: Optional[str] = None,
    history_json: str = Form("[]"),
    files: List[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files received.")

    if len(files) > MAX_ATTACHMENTS:
        raise limit_error(
            "TOO_MANY_ATTACHMENTS",
            f"You can attach up to {MAX_ATTACHMENTS} files to a message.",
            limit=MAX_ATTACHMENTS, attempted=len(files),
        )

    attachments, oversized, wrong_type = [], [], []
    total = 0

    for f in files:
        mime = (f.content_type or "").split(";")[0].strip().lower()
        if mime not in ALLOWED_ATTACHMENT_TYPES:
            wrong_type.append(f"{f.filename} — {mime or 'unknown type'}")
            continue
        data = await f.read()
        total += len(data)
        if len(data) > MAX_ATTACHMENT_MB * MB:
            oversized.append({"filename": f.filename, "size_mb": round(len(data) / MB, 2)})
            continue
        attachments.append({"filename": f.filename, "mime_type": mime, "data": data})

    if wrong_type:
        raise HTTPException(status_code=400, detail={
            "code": "INVALID_ATTACHMENT",
            "message": "Only images (JPEG, PNG, WebP, HEIC) and PDFs can be attached.",
            "files": wrong_type,
        })

    if oversized:
        raise limit_error(
            "ATTACHMENT_TOO_LARGE",
            f"Each attachment must be {MAX_ATTACHMENT_MB:g} MB or smaller.",
            limit_mb=MAX_ATTACHMENT_MB, files=oversized,
        )

    if not attachments:
        raise HTTPException(status_code=400, detail="No usable attachments.")

    try:
        history = [
            t.model_dump() for t in AskRequest(**{"history": json.loads(history_json)}).history
            if t.content
        ][-MAX_HISTORY_TURNS:]
    except Exception:
        history = []

    doc_context = ""
    if query.strip():
        doc_context = await run_in_threadpool(
            rag_service.build_context_for_query,
            current_user["user_id"], query, 6, upload_id
        )

    result = await run_in_threadpool(
        rag_service.answer_with_attachments,
        query, attachments, history, doc_context
    )
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("message", "Failed to read attachments"))
    return result


@app.post("/ask")
async def ask_question(
    query: str,
    top_k: int = 12,
    upload_id: Optional[str] = None,
    payload: Optional[AskRequest] = None,
    current_user: dict = Depends(get_current_user)
):
    try:
        if not query or len(query.strip()) < 1:
            raise HTTPException(status_code=400, detail="Please enter a message.")

        turns = payload.history if payload else []
        history = [t.model_dump() for t in turns if t.content][-MAX_HISTORY_TURNS:]

        result = await run_in_threadpool(
            rag_service.generate_answer,
            current_user["user_id"], query, top_k, upload_id, history
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
        doc = documents_collection.find_one(
            {"user_id": current_user["user_id"], "upload_id": upload_id},
            {"text_cloudinary_url": 1}
        )
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        return await run_in_threadpool(
            rag_service.index_document,
            current_user["user_id"], upload_id, doc["text_cloudinary_url"]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reindex-all")
async def reindex_all_documents(current_user: dict = Depends(get_current_user)):
    try:
        docs = list(documents_collection.aggregate([
            {"$match": {"user_id": current_user["user_id"]}},
            {"$group": {"_id": "$upload_id", "text_cloudinary_url": {"$first": "$text_cloudinary_url"}}}
        ]))
        if not docs:
            return {"success": True, "message": "No documents to index", "indexed_documents": 0}

        indexed, errors = 0, []
        for d in docs:
            res = await run_in_threadpool(
                rag_service.index_document,
                current_user["user_id"], d["_id"], d.get("text_cloudinary_url")
            )
            if res["success"]:
                indexed += 1
            else:
                errors.append({"upload_id": d["_id"], "error": res.get("message")})

        return {
            "success": True,
            "total_documents": len(docs),
            "indexed_documents": indexed,
            "failed_documents": len(errors),
            "errors": errors or None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Voice endpoints ──────────────────────────────────────────────────────────

class SpeakRequest(BaseModel):
    text: str


@app.post("/speak-chunks")
async def speak_chunks(
    request: SpeakRequest,
    current_user: dict = Depends(get_current_user),
):
    """Splits an answer into speakable segments. The client requests audio for
    one segment at a time so no single response approaches the 4.5 MB ceiling."""
    chunks = voice_service.split_for_speech(request.text)
    return {"success": True, "chunks": chunks, "count": len(chunks)}


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file received.")

        mime_type = file.content_type or "audio/webm"
        mime_type = mime_type.split(";")[0].strip()

        text = voice_service.transcribe(audio_bytes, mime_type=mime_type)
        return {"success": True, "text": text}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/speak")
async def speak_text(
    request: SpeakRequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="No text provided.")

        wav_bytes = voice_service.synthesize(request.text)
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": "inline; filename=acumen-response.wav"},
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)