"""
Verifies that signed + authenticated + raw delivery works on THIS Cloudinary account,
and — more importantly — that unsigned access is actually refused.

Run from the same directory as your .env:
    python verify_cloudinary.py
"""

import os
import sys
import re
import tempfile

import requests
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader
import cloudinary.utils

load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True,
)

if not cloudinary.config().cloud_name:
    sys.exit("CLOUDINARY_* variables not found. Run this next to your .env file.")

PUBLIC_ID = "acumen/_verification/signed-delivery-probe.pdf"
GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"

results = []


def record(name: str, passed: bool, detail: str) -> None:
    results.append((name, passed, detail))
    mark = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
    print(f"  [{mark}] {name}\n         {DIM}{detail}{RESET}")


def make_pdf() -> str:
    path = os.path.join(tempfile.gettempdir(), "acumen_probe.pdf")
    try:
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 100), "Acumen Cloudinary delivery probe")
        doc.save(path)
        doc.close()
    except ImportError:
        with open(path, "wb") as f:
            f.write(
                b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
                b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
                b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>endobj\n"
                b"trailer<</Root 1 0 R>>\n%%EOF\n"
            )
    return path


def get(url: str):
    try:
        return requests.get(url, timeout=20, allow_redirects=True)
    except Exception as e:
        return e


print(f"\n{'=' * 66}\n Cloudinary signed-delivery verification\n cloud: {cloudinary.config().cloud_name}\n{'=' * 66}\n")

pdf_path = make_pdf()

print("Uploading probe as resource_type=raw, type=authenticated ...\n")
try:
    upload = cloudinary.uploader.upload(
        pdf_path,
        resource_type="raw",
        type="authenticated",
        public_id=PUBLIC_ID,
        overwrite=True,
    )
except Exception as e:
    sys.exit(f"{RED}Upload failed:{RESET} {e}\n"
             f"Authenticated uploads may be disabled on this account.")

print(f"{DIM}stored public_id: {upload['public_id']}{RESET}\n")

signed_url, _ = cloudinary.utils.cloudinary_url(
    PUBLIC_ID, resource_type="raw", type="authenticated", sign_url=True, secure=True
)
unsigned_url = re.sub(r"/s--[^/]+--/", "/", signed_url)
public_url = signed_url.replace("/raw/authenticated/", "/raw/upload/")
public_url = re.sub(r"/s--[^/]+--/", "/", public_url)

print("Checks:\n")

# 1. the signed URL your /view endpoint hands out must work
r = get(signed_url)
if isinstance(r, Exception):
    record("Signed URL is fetchable", False, f"request error: {r}")
else:
    record("Signed URL is fetchable", r.status_code == 200,
           f"HTTP {r.status_code} — {signed_url[:88]}")

    ctype = r.headers.get("Content-Type", "")
    disp = r.headers.get("Content-Disposition", "")
    inline = "pdf" in ctype.lower() and "attachment" not in disp.lower()
    record("Renders inline in a browser tab", inline,
           f"Content-Type: {ctype or 'none'} | Content-Disposition: {disp or 'none'}")

# 2. THE IMPORTANT ONE — no signature must mean no access
r = get(unsigned_url)
if isinstance(r, Exception):
    record("Unsigned URL is refused", False, f"request error: {r}")
else:
    record("Unsigned URL is refused", r.status_code in (401, 403, 404),
           f"HTTP {r.status_code} — expected 401/403/404")

# 3. the same path under the public delivery type must not resolve
r = get(public_url)
if isinstance(r, Exception):
    record("Not readable via public delivery type", False, f"request error: {r}")
else:
    record("Not readable via public delivery type", r.status_code in (401, 403, 404),
           f"HTTP {r.status_code} — expected 401/403/404")

# 4. fallback option, in case signed delivery is unavailable on this plan
try:
    dl = cloudinary.utils.private_download_url(
        PUBLIC_ID, None, resource_type="raw", type="authenticated", expires_at=None
    )
    r = get(dl)
    ok = (not isinstance(r, Exception)) and r.status_code == 200
    record("Fallback: private_download_url works", ok,
           f"HTTP {getattr(r, 'status_code', r)} — use this if signed delivery fails")
except Exception as e:
    record("Fallback: private_download_url works", False, str(e))

print(f"\n{DIM}Cleaning up probe asset ...{RESET}")
try:
    cloudinary.uploader.destroy(PUBLIC_ID, resource_type="raw", type="authenticated")
except Exception as e:
    print(f"{YELLOW}Could not delete probe, remove {PUBLIC_ID} manually: {e}{RESET}")

blocking = [n for n, p, _ in results if not p and n in (
    "Signed URL is fetchable", "Unsigned URL is refused",
    "Not readable via public delivery type")]

print(f"\n{'=' * 66}")
if not blocking:
    print(f"{GREEN}Good to go.{RESET} Signed delivery works and unsigned access is refused.")
elif "Unsigned URL is refused" in blocking or "Not readable via public delivery type" in blocking:
    print(f"{RED}STOP — your documents are publicly readable.{RESET}")
    print("Anyone holding the URL can read any user's PDF. Do not ship this.")
    print("Fix: Cloudinary Console > Settings > Security > enable strict access")
    print("for authenticated/raw assets, then re-run.")
else:
    print(f"{YELLOW}Signed delivery is not working on this account.{RESET}")
    print("Switch signed_pdf_url() in main.py to private_download_url() —")
    print("check whether that fallback passed above.")
print(f"{'=' * 66}\n")