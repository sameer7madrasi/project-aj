import os
import sys
import io
import base64
from typing import List

from dotenv import load_dotenv
from PIL import Image
from openai import OpenAI

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

# Try to import pillow-heif for HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HAS_HEIF = True
except ImportError:
    HAS_HEIF = False

# 1. Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set in .env")

client = OpenAI(api_key=api_key)


def file_to_images(file_path: str, dpi: int = 300) -> List[Image.Image]:
    """
    Convert a PDF or image file into a list of PIL Images.
    Supports PDF, HEIC, JPEG, PNG, and other image formats.
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    # Handle PDF files
    if ext == '.pdf':
        if not HAS_PYMUPDF:
            raise RuntimeError("PyMuPDF (fitz) is required for PDF files. Install it with: pip install PyMuPDF")
        
        doc = fitz.open(file_path)
        images = []
        
        # Calculate zoom factor for desired DPI (default PDF DPI is 72)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Render page to pixmap
            pix = page.get_pixmap(matrix=mat)
            # Convert to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        
        doc.close()
        return images
    
    # Handle image files (HEIC, JPEG, PNG, etc.)
    else:
        try:
            img = Image.open(file_path)
            # Convert to RGB if necessary (HEIC might be in other modes)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return [img]
        except Exception as e:
            if ext in ['.heic', '.heif'] and not HAS_HEIF:
                raise RuntimeError(
                    f"HEIC/HEIF format detected but pillow-heif is not installed.\n"
                    f"Install it with: pip install pillow-heif"
                ) from e
            raise RuntimeError(f"Failed to open image file: {e}") from e


def pil_image_to_data_uri(img: Image.Image, fmt: str = "JPEG") -> str:
    """
    Convert a PIL image to a base64 data URI that OpenAI's vision models can read.
    """
    buffer = io.BytesIO()
    img.save(buffer, format=fmt)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    mime = "image/jpeg" if fmt.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"


def ocr_page_with_openai(img: Image.Image) -> str:
    """
    Send a single page image to OpenAI's vision model and return the extracted text.
    """
    image_data_uri = pil_image_to_data_uri(img, fmt="JPEG")

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "This is a scanned handwritten diary page.\n\n" +
              "Transcribe the handwriting as accurately as possible into plain text. " +
              "Preserve the original wording and approximate line breaks.\n\n" +
              "If any word or phrase is unclear or illegible, DO NOT guess. " +
              'Instead, insert the token "<illegible>" in place of that word or phrase.\n\n' +
              "Do not add commentary, explanations, or summaries. " +
              "Only output the transcribed diary content."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": image_data_uri,
                    },
                ],
            }
        ],
    )

    # Simplest extraction of text content
    try:
        # If your SDK version supports this convenience:
        return response.output_text
    except AttributeError:
        # Fallback: dig into the first output content item
        first_output = response.output[0]
        first_content = first_output.content[0]
        return first_content.text


def ocr_file_with_openai(file_path: str) -> str:
    """
    Convert a PDF or image file to text using OpenAI vision, one page at a time.
    """
    images = file_to_images(file_path)
    all_text = []

    for i, img in enumerate(images):
        print(f"Processing page {i + 1}/{len(images)} with OpenAI Vision...")
        page_text = ocr_page_with_openai(img)
        all_text.append(f"=== PAGE {i + 1} ===\n{page_text.strip()}\n")

    return "\n\n".join(all_text)


def main():
    if len(sys.argv) < 2:
        print("Usage: python ocr.py input/YourFile.pdf")
        print("       python ocr.py input/YourImage.jpg")
        print("       python ocr.py input/YourImage.HEIC")
        print("\nOr use the helper script: ./ocr.sh input/YourFile.pdf")
        sys.exit(1)

    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print("File not found:", file_path)
        sys.exit(1)

    print(f"Running OCR on: {file_path}")
    text = ocr_file_with_openai(file_path)

    # Save to output/
    os.makedirs("output", exist_ok=True)
    base_name = os.path.basename(file_path)
    name_without_ext = os.path.splitext(base_name)[0]
    out_path = os.path.join("output", f"{name_without_ext}_openai.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    print("\nDone.")
    print(f"Output saved to: {out_path}")


if __name__ == "__main__":
    main()
