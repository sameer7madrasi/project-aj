import os
from typing import List, Any

from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("Missing Supabase config in .env")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def get_pages_without_embeddings(limit: int = 50) -> List[dict]:
    """
    Fetch diary_pages rows where embedding is null.
    """
    # Supabase Python client: .is_("embedding", "null") won't work; we use raw filter
    resp = (
        supabase.table("diary_pages")
        .select("*")
        .filter("embedding", "is", "null")
        .limit(limit)
        .execute()
    )
    return resp.data or []


def create_embedding(text: str) -> list[float]:
    """
    Call OpenAI embeddings API for the given text.
    """
    # Optionally truncate very long text if needed
    text = text.strip()
    if not text:
        return []

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


def update_page_embedding(page_id: str, embedding: List[float]) -> Any:
    """
    Update a single diary_pages row with the given embedding.
    """
    resp = (
        supabase.table("diary_pages")
        .update({"embedding": embedding})
        .eq("id", page_id)
        .execute()
    )
    return resp


def main():
    print("Fetching pages without embeddings...")
    pages = get_pages_without_embeddings(limit=50)

    if not pages:
        print("No pages found without embeddings. All caught up!")
        return

    print(f"Found {len(pages)} pages to embed.")

    for page in pages:
        page_id = page["id"]
        text = page.get("clean_text") or page.get("raw_text") or ""
        print(f"\nEmbedding page {page_id[:8]}...")

        if not text.strip():
            print("  Skipping: empty text.")
            continue

        embedding = create_embedding(text)
        print(f"  Embedding length: {len(embedding)}")

        update_page_embedding(page_id, embedding)
        print("  Updated embedding in Supabase.")

    print("\nDone backfilling embeddings for this batch.")


if __name__ == "__main__":
    main()

