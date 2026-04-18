import fitz  # PyMuPDF
import json
import re
import os
import requests
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
import bcrypt
from jose import JWTError, jwt
from pydantic import BaseModel, field_validator
from bson import ObjectId
import cloudinary
import cloudinary.uploader

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

# ─── Cloudinary configuration ─────────────────────────────────────────────────
# Add these three lines to your .env file — get the values from
# cloudinary.com/console → Settings → Access Keys:
#
#   CLOUDINARY_CLOUD_NAME=your_cloud_name
#   CLOUDINARY_API_KEY=your_api_key
#   CLOUDINARY_API_SECRET=your_api_secret
#
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True,
)

# ─── MongoDB client ───────────────────────────────────────────────────────────
context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
context.minimum_version = ssl.TLSVersion.TLSv1_2

client = MongoClient(MONGO_URI, tls=True, tlsAllowInvalidCertificates=True)
db = client[DATABASE_NAME]
users_collection = db[USERS_COLLECTION]
documents_collection = db[DOCUMENTS_COLLECTION]

# ─── RAG Service ──────────────────────────────────────────────────────────────
rag_service = get_rag_service()

# MongoDB indexes
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


# ─── Password helpers ─────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, stored_hash: str, user_email: str) -> bool:
    # Legacy SHA-256 migration shim
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

def upload_text_to_cloudinary(sections_data: list, user_id: str, upload_id: str) -> dict:
    """
    Serialise extracted text sections to JSON and store them in Cloudinary.

    This keeps MongoDB documents tiny (metadata only, ~1KB each) while
    the actual text content (potentially hundreds of KB per document) lives
    in Cloudinary's cheap object storage.
    """
    payload = json.dumps({
        "upload_id": upload_id,
        "user_id": user_id,
        "created_at": datetime.utcnow().isoformat(),
        "sections": sections_data,
    }, ensure_ascii=False).encode("utf-8")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp.write(payload)
        tmp_path = tmp.name

    try:
        result = cloudinary.uploader.upload(
            tmp_path,
            resource_type="raw",
            folder=f"acumen/{user_id}/text",
            public_id=f"{upload_id}_sections",
            overwrite=True,
        )
    finally:
        os.unlink(tmp_path)

    return {
        "url": result["secure_url"],
        "public_id": result["public_id"],
        "bytes": result.get("bytes", 0),
    }


def delete_cloudinary_resources(public_ids: list) -> None:
    """Delete Cloudinary resources by public_id (best-effort, non-fatal)."""
    for pid in public_ids:
        try:
            cloudinary.uploader.destroy(pid, resource_type="raw")
        except Exception as e:
            print(f"Cloudinary delete failed for {pid}: {e}")


def fetch_sections_from_cloudinary(text_url: str) -> list:
    """Fetch and parse the sections JSON stored in Cloudinary."""
    try:
        resp = requests.get(text_url, timeout=15)
        resp.raise_for_status()
        return resp.json().get("sections", [])
    except Exception as e:
        print(f"Failed to fetch sections from Cloudinary ({text_url}): {e}")
        return []


# ─── PDF text extraction ──────────────────────────────────────────────────────

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
            text = "\n".join(b[4] for b in blocks if len(b) >= 5 and b[4].strip())
        except Exception:
            pass
    if not text or len(text.strip()) < 50:
        try:
            d = page.get_text("dict")
            parts = []
            for block in d.get("blocks", []):
                if block.get("type") == 0:
                    for line in block.get("lines", []):
                        t = "".join(s.get("text", "") for s in line.get("spans", []))
                        if t.strip():
                            parts.append(t)
            text = "\n".join(parts)
        except Exception:
            pass
    return clean_text(text)


def extract_sections_advanced(doc) -> list:
    full_text = ""
    page_texts = []
    for page_num, page in enumerate(doc):
        t = extract_text_from_page(page)
        if t:
            page_texts.append({"page_number": page_num + 1, "text": t, "char_count": len(t)})
            full_text += t + "\n\n"

    if not full_text.strip():
        raise Exception("No text could be extracted. The PDF may be image-based or encrypted.")

    patterns = [
        r'^(?:Chapter|CHAPTER|Ch\.|CH\.)\s+(\d+|[IVXLCDM]+)[\:\.\s]+(.+?)$',
        r'^(?:Section|SECTION|Sec\.|SEC\.)\s+(\d+|[IVXLCDM]+)[\:\.\s]+(.+?)$',
        r'^(\d+)\.\s+([A-Z][^\n]{10,100})$',
        r'^([A-Z][A-Z\s]{3,80})$',
        r'^(\d+\.\d+)\s+(.+?)$',
        r'^(?:Part|PART|Book|BOOK)\s+(\d+|[IVXLCDM]+)[\:\.\s]+(.+?)$',
    ]
    detected = []
    for p in patterns:
        m = list(re.finditer(p, full_text, re.MULTILINE))
        if len(m) >= 2:
            detected = m
            break

    cpp = len(full_text) / len(doc) if len(doc) > 0 else 1000
    sections = []

    if detected:
        for i, match in enumerate(detected):
            title = match.group(0).strip()
            start_pos = match.start()
            end_pos = detected[i + 1].start() if i + 1 < len(detected) else len(full_text)
            text = full_text[start_pos:end_pos].strip()
            sp = max(1, int(len(full_text[:start_pos]) / cpp) + 1)
            ep = min(len(doc), sp + max(1, int(len(text) / cpp)))
            sections.append({
                "section_title": title, "start_page": sp, "end_page": ep,
                "text_content": text, "section_type": "detected", "char_count": len(text),
            })
    elif len(doc) <= 5:
        sections.append({
            "section_title": "Complete Document", "start_page": 1, "end_page": len(doc),
            "text_content": full_text, "section_type": "full_document", "char_count": len(full_text),
        })
    else:
        for i in range(0, len(page_texts), 6):
            chunk = page_texts[i:i + 6]
            sp, ep = chunk[0]["page_number"], chunk[-1]["page_number"]
            text = "\n\n".join(p["text"] for p in chunk)
            title = next(
                (l.strip() for l in text.split("\n")[:10] if 10 <= len(l.strip()) <= 100 and not l.strip().endswith(".")),
                f"Pages {sp}-{ep}"
            )
            sections.append({
                "section_title": title, "start_page": sp, "end_page": ep,
                "text_content": text, "section_type": "page_chunk", "char_count": len(text),
            })

    return sections


def process_pdf(pdf_path: str, user_id: str, filename: str, upload_id: str) -> dict:
    """
    Core processing pipeline:
      1. Extract text sections from the PDF
      2. Upload sections JSON to Cloudinary  (text storage — keeps MongoDB lean)
      3. Save lightweight metadata to MongoDB  (NO text_content field)
    """
    doc = fitz.open(pdf_path)
    print(f"\nProcessing: {filename} ({len(doc)} pages)")
    sections_data = extract_sections_advanced(doc)
    metadata = doc.metadata
    total_pages = len(doc)
    doc.close()

    if not sections_data:
        raise Exception("No content extracted from PDF")

    print("Uploading extracted text to Cloudinary...")
    text_cl = upload_text_to_cloudinary(sections_data, user_id, upload_id)
    print(f"Text uploaded ({text_cl['bytes']:,} bytes): {text_cl['url']}")

    # MongoDB records — metadata only, no text_content
    records = []
    for idx, section in enumerate(sections_data):
        records.append({
            "user_id": user_id,
            "upload_id": upload_id,
            "filename": filename,
            "section_number": idx + 1,
            "section_title": section["section_title"],
            "section_type": section["section_type"],
            "char_count": section["char_count"],       # size indicator, not the text itself
            "start_page": section["start_page"],
            "end_page": section["end_page"],
            "total_pages": total_pages,
            "pdf_metadata": {
                "author": metadata.get("author", ""),
                "title": metadata.get("title", ""),
                "subject": metadata.get("subject", ""),
            },
            # Cloudinary text reference (same for all sections of this upload)
            "text_cloudinary_url": text_cl["url"],
            "text_cloudinary_public_id": text_cl["public_id"],
            "uploaded_at": datetime.utcnow(),
        })

    result = documents_collection.insert_many(records)
    print(f"Saved {len(result.inserted_ids)} metadata records to MongoDB")

    return {
        "success": True,
        "upload_id": upload_id,
        "sections_inserted": len(result.inserted_ids),
        "total_pages": total_pages,
        "total_characters": sum(s["char_count"] for s in sections_data),
        "text_cloudinary_url": text_cl["url"],
        "message": "PDF processed successfully",
    }


# ─── Auth endpoints ───────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message": "ACUMEN PDF API",
        "version": "5.0",
        "storage": "MongoDB (metadata) + Cloudinary (text) + Pinecone (vectors)",
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
        if not user or not verify_password(request.password, user["password_hash"], request.email):
            raise HTTPException(status_code=401, detail="Invalid email or password.")
        token = create_access_token(user_id=user["user_id"], email=user["email"])
        users_collection.update_one({"email": request.email}, {"$set": {"last_login": datetime.utcnow()}})
        return {"success": True, "user_token": token, "user_id": user["user_id"],
                "name": user["name"], "email": user["email"], "message": "Login successful."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Protected endpoints ──────────────────────────────────────────────────────

@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    upload_id = secrets.token_urlsafe(16)
    tmp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(await file.read())
            tmp_file_path = tmp.name

        result = process_pdf(
            pdf_path=tmp_file_path,
            user_id=current_user["user_id"],
            filename=file.filename,
            upload_id=upload_id,
        )
        os.unlink(tmp_file_path)
        tmp_file_path = None

        if result["success"]:
            index_result = rag_service.index_document(
                user_id=current_user["user_id"],
                upload_id=upload_id,
                text_cloudinary_url=result["text_cloudinary_url"],
            )
            result["rag_indexing"] = index_result
            result["rag_indexed"] = index_result.get("success", False)
            if not result["rag_indexed"]:
                result["rag_indexing_warning"] = (
                    "Document saved but AI indexing failed. "
                    "ACUMEN won't recognise this document yet. "
                    "Use the Re-index button to retry."
                )

        return result

    except Exception as e:
        if tmp_file_path and os.path.exists(tmp_file_path):
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
                "pdf_title": {"$first": "$pdf_metadata.title"},
                "text_cloudinary_url": {"$first": "$text_cloudinary_url"},
            }},
            {"$sort": {"uploaded_at": -1}},
        ]
        return {
            "success": True,
            "user_id": current_user["user_id"],
            "total_uploads": len(docs := list(documents_collection.aggregate(pipeline))),
            "documents": docs,
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
            "text_cloudinary_url": sections[0].get("text_cloudinary_url"),
            "sections": sections,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/document/{upload_id}")
async def delete_document(upload_id: str, current_user: dict = Depends(get_current_user)):
    try:
        sample = documents_collection.find_one(
            {"user_id": current_user["user_id"], "upload_id": upload_id},
            {"text_cloudinary_public_id": 1}
        )
        if not sample:
            raise HTTPException(status_code=404, detail="Document not found")

        delete_cloudinary_resources([
            pid for pid in [
                sample.get("text_cloudinary_public_id"),
            ] if pid
        ])

        db_result = documents_collection.delete_many(
            {"user_id": current_user["user_id"], "upload_id": upload_id}
        )
        rag_result = rag_service.delete_document_vectors(current_user["user_id"], upload_id)

        return {
            "success": True,
            "deleted_sections": db_result.deleted_count,
            "rag_deletion": rag_result,
            "message": "Document deleted from all storage layers",
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

        enhanced = []
        for r in results["results"]:
            try:
                doc = documents_collection.find_one(
                    {"_id": ObjectId(r["original_doc_ref"])},
                    {"pdf_metadata": 1, "start_page": 1, "end_page": 1}
                )
                if doc:
                    r["pdf_metadata"] = doc.get("pdf_metadata", {})
                    r["start_page"] = doc.get("start_page")
                    r["end_page"] = doc.get("end_page")
            except Exception:
                pass
            enhanced.append(r)
        results["results"] = enhanced
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
    """Re-index a single document. Call this if ACUMEN can't find an uploaded document."""
    try:
        doc = documents_collection.find_one(
            {"user_id": current_user["user_id"], "upload_id": upload_id},
            {"text_cloudinary_url": 1}
        )
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        return rag_service.index_document(
            user_id=current_user["user_id"],
            upload_id=upload_id,
            text_cloudinary_url=doc["text_cloudinary_url"],
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reindex-all")
async def reindex_all_documents(current_user: dict = Depends(get_current_user)):
    """Re-index all documents. Run this once after deploying the Cloudinary migration."""
    try:
        docs = list(documents_collection.aggregate([
            {"$match": {"user_id": current_user["user_id"]}},
            {"$group": {"_id": "$upload_id", "text_cloudinary_url": {"$first": "$text_cloudinary_url"}}}
        ]))
        if not docs:
            return {"success": True, "message": "No documents to index", "indexed_documents": 0}

        indexed, errors = 0, []
        for d in docs:
            res = rag_service.index_document(
                user_id=current_user["user_id"],
                upload_id=d["_id"],
                text_cloudinary_url=d.get("text_cloudinary_url"),
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)