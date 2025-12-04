import os
import sys
from datetime import date
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
USER_ID = os.getenv("PROJECTAJ_USER_ID")
MAIN_DIARY_ID = os.getenv("PROJECTAJ_MAIN_DIARY_ID")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("Missing Supabase URL or service role key in .env")

if not USER_ID or not MAIN_DIARY_ID:
    raise RuntimeError("Missing PROJECTAJ_USER_ID or PROJECTAJ_MAIN_DIARY_ID in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def ingest_text_file(txt_path: str, entry_date: date | None = None, page_number: int | None = None):
    # Read the file
    with open(txt_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # For now, we'll just set clean_text = raw_text
    clean_text = raw_text

    source_file_name = os.path.basename(txt_path)

    payload = {
        "user_id": USER_ID,
        "diary_id": MAIN_DIARY_ID,
        "page_number": page_number,
        "source_file_name": source_file_name,
        "image_path": None,  # later: link to Supabase Storage file
        "raw_text": raw_text,
        "clean_text": clean_text,
        "entry_date": entry_date.isoformat() if entry_date else None,
    }

    print("Inserting row into diary_pages...")
    res = supabase.table("diary_pages").insert(payload).execute()

    print("Insert result:", res)


def main():
    if len(sys.argv) < 2:
        print("Usage: python ingest_to_supabase.py path/to/output.txt [YYYY-MM-DD] [page_number]")
        sys.exit(1)

    txt_path = sys.argv[1]
    if not os.path.exists(txt_path):
        print("Text file not found:", txt_path)
        sys.exit(1)

    entry_date = None
    if len(sys.argv) >= 3 and sys.argv[2] != "None":
        entry_date = date.fromisoformat(sys.argv[2])

    page_number = None
    if len(sys.argv) >= 4 and sys.argv[3] != "None":
        page_number = int(sys.argv[3])

    ingest_text_file(txt_path, entry_date=entry_date, page_number=page_number)


if __name__ == "__main__":
    main()

