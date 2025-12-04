import os
import re
import sys
from datetime import date, datetime

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


def infer_entry_date(text: str) -> date | None:
    """
    Try to infer a date from the OCR text.

    Strategy (no external deps, stdlib only):
    - Look at the first few non-empty lines for:
      * ISO dates: YYYY-MM-DD
      * Month-name dates: e.g. "Jan 1st, 2024", "January 1, 2024"
    - Return a `date` if something parses, otherwise None.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    lines = lines[:10]  # only inspect first few lines

    # 1) ISO format: YYYY-MM-DD
    iso_re = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
    for line in lines:
        m = iso_re.search(line)
        if m:
            try:
                return date.fromisoformat(m.group(1))
            except ValueError:
                pass

    # 2) Month-name formats, e.g. "Jan 1st, 2024" or "January 1, 2024"
    month_re = re.compile(
        r"\b("
        r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|"
        r"Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2})(?:st|nd|rd|th)?\s*,\s*(\d{4})",
        re.IGNORECASE,
    )
    for line in lines:
        m = month_re.search(line)
        if m:
            month_str, day_str, year_str = m.groups()
            # Normalize e.g. "jan" -> "Jan"
            month_norm = month_str[:1].upper() + month_str[1:].lower()
            try:
                dt = datetime.strptime(
                    f"{month_norm} {int(day_str)}, {year_str}", "%b %d, %Y"
                )
                return dt.date()
            except ValueError:
                try:
                    dt = datetime.strptime(
                        f"{month_norm} {int(day_str)}, {year_str}", "%B %d, %Y"
                    )
                    return dt.date()
                except ValueError:
                    continue

    return None


def infer_page_number(text: str) -> int | None:
    """
    Try to infer the page number from OCR text.

    For our OCR output, pages are typically marked as:
        === PAGE 1 ===
    at the top of each page.
    """
    for line in text.splitlines():
        m = re.match(r"^===\s*PAGE\s+(\d+)\s*===\s*$", line.strip(), re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return None
    return None


def ingest_text_file(txt_path: str, entry_date: date | None = None, page_number: int | None = None):
    """
    Ingest a single OCR output text file into Supabase.

    - If `entry_date` is not provided, we try to infer it from the text.
    - If `page_number` is not provided, we try to infer it from the text
      (e.g., from a leading \"=== PAGE N ===\" marker).
    - Both fields are optional so this works for non-diary content too
      (recipes, notes, exams, etc.).
    """
    # Read the file
    with open(txt_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Infer entry_date if not provided
    if entry_date is None:
        inferred_date = infer_entry_date(raw_text)
        if inferred_date:
            print(f"Inferred entry_date from text: {inferred_date}")
            entry_date = inferred_date
        else:
            print("No date inferred from text; leaving entry_date as NULL.")

    # Infer page_number if not provided
    if page_number is None:
        inferred_page = infer_page_number(raw_text)
        if inferred_page is not None:
            print(f"Inferred page_number from text: {inferred_page}")
            page_number = inferred_page
        else:
            print("No page number inferred from text; leaving page_number as NULL.")

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
        print("If you omit date/page, the script will try to infer them from the OCR text.")
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

