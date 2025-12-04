#!/bin/bash
# Helper script to run OCR with virtual environment activated

cd "$(dirname "$0")"
source .venv/bin/activate
python ocr.py "$@"

