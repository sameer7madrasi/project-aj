#!/usr/bin/env python3
"""
Batch ingest all OCR output files from the output/ directory into Supabase.
Then backfill embeddings for all ingested records.
"""
import os
import sys
from pathlib import Path

# Import the ingestion function
from ingest_to_supabase import ingest_text_file

def main():
    output_dir = Path("output")
    
    if not output_dir.exists():
        print(f"Error: {output_dir} directory not found.")
        sys.exit(1)
    
    # Find all .txt files in output directory
    txt_files = sorted(output_dir.glob("*.txt"))
    
    if not txt_files:
        print(f"No .txt files found in {output_dir}")
        sys.exit(0)
    
    print(f"Found {len(txt_files)} files to ingest:\n")
    for f in txt_files:
        print(f"  - {f.name}")
    
    print("\n" + "="*60)
    print("Starting batch ingestion...")
    print("="*60 + "\n")
    
    success_count = 0
    error_count = 0
    
    for i, txt_file in enumerate(txt_files, 1):
        print(f"\n[{i}/{len(txt_files)}] Processing: {txt_file.name}")
        print("-" * 60)
        
        try:
            # Ingest with automatic date/page inference
            ingest_text_file(str(txt_file))
            success_count += 1
            print(f"✓ Successfully ingested: {txt_file.name}")
        except Exception as e:
            error_count += 1
            print(f"✗ Error ingesting {txt_file.name}: {e}")
            # Continue with next file
            continue
    
    print("\n" + "="*60)
    print("Batch ingestion complete!")
    print(f"  Success: {success_count}")
    print(f"  Errors: {error_count}")
    print("="*60)
    
    if success_count > 0:
        print("\nNow running backfill_embeddings.py to generate embeddings...")
        print("="*60 + "\n")
        
        # Import and run backfill
        from backfill_embeddings import main as backfill_main
        backfill_main()
    
    print("\n✓ All done!")


if __name__ == "__main__":
    main()

