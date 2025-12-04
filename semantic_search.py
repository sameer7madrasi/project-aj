import os
import sys
from typing import List

from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_ANON_KEY in .env")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


def create_query_embedding(query: str) -> List[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query,
    )
    return response.data[0].embedding


def semantic_search(query: str, match_count: int = 5):
    embedding = create_query_embedding(query)

    # Call the Postgres function we created
    resp = supabase.rpc(
        "match_diary_pages",
        {
            "query_embedding": embedding,
            "match_count": match_count,
        },
    ).execute()

    return resp.data or []


def pretty_print_results(query: str, results: list[dict]):
    print(f"\nQuery: {query}\n")
    if not results:
        print("No results found.")
        return

    for i, row in enumerate(results, start=1):
        similarity = row.get("similarity", 0)
        entry_date = row.get("entry_date")
        page_number = row.get("page_number")
        snippet = (row.get("clean_text") or row.get("raw_text") or "").strip()

        if len(snippet) > 300:
            snippet = snippet[:300] + "..."

        print(f"Result {i}:")
        print(f"  Similarity: {similarity:.3f}")
        print(f"  Date: {entry_date} | Page: {page_number}")
        print("  Snippet:")
        print(f"    {snippet}")
        print("-" * 60)


def main():
    if len(sys.argv) < 2:
        print("Usage: python semantic_search.py \"your search query\" [match_count]")
        sys.exit(1)

    query = sys.argv[1]
    match_count = int(sys.argv[2]) if len(sys.argv) >= 3 else 5

    results = semantic_search(query, match_count=match_count)
    pretty_print_results(query, results)


if __name__ == "__main__":
    main()

