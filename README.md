# OCR Project

A Python-based OCR (Optical Character Recognition) tool that uses OpenAI's Vision API to extract text from PDFs and images, including handwritten content.

## Features

- ✅ **Multi-format support**: PDFs, HEIC, JPEG, PNG, and other image formats
- ✅ **Handwriting recognition**: Excellent accuracy for handwritten text
- ✅ **Multi-page PDFs**: Automatically processes all pages
- ✅ **High quality**: Uses OpenAI's GPT-4.1-mini vision model

## Setup

1. **Create a virtual environment** (if not already created):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install openai python-dotenv PyMuPDF pillow pillow-heif
   ```

3. **Set up your OpenAI API key**:
   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

### Method 1: Using the helper script (recommended)
```bash
./ocr.sh "input/YourFile.pdf"
./ocr.sh "input/YourImage.jpg"
./ocr.sh "input/YourImage.HEIC"
```

### Method 2: Direct Python execution
```bash
source .venv/bin/activate
python ocr.py "input/YourFile.pdf"
```

## Supported File Formats

- **PDFs**: `.pdf` (all pages processed)
- **Images**: `.jpg`, `.jpeg`, `.png`, `.heic`, `.heif`, and other formats supported by PIL

## Output

Processed text files are saved to the `output/` directory with the naming pattern:
`{filename}_openai.txt`

## Project Structure

```
ProjectAJ/
├── ocr.py              # Main OCR script
├── ocr.sh              # Helper script (auto-activates venv)
├── input/              # Place your PDFs/images here
├── output/             # Processed text files appear here
├── .env                # Your OpenAI API key (not in git)
└── README.md           # This file
```

## Requirements

- Python 3.8+
- OpenAI API key with billing enabled
- Virtual environment (recommended)

## Notes

- The script processes files page by page for multi-page PDFs
- Each page is clearly marked in the output with `=== PAGE N ===`
- Processing time depends on file size and number of pages
- Costs are based on OpenAI API usage

