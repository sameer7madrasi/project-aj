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
    raise RuntimeError("PyMuPDF (fitz) is required. Install it with: pip install PyMuPDF")

# 1. Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set in .env")

client = OpenAI(api_key=api_key)


def pdf_to_images(pdf_path: str, dpi: int = 300) -> List[Image.Image]:
    """
    Convert a PDF into a list of PIL Images, one per page, using PyMuPDF.
    """
    if not HAS_PYMUPDF:
        raise RuntimeError("PyMuPDF (fitz) is required. Install it with: pip install PyMuPDF")
    
    doc = fitz.open(pdf_path)
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
                            "This is a scanned handwritten diary page. "
                            "Please transcribe the handwriting as accurately as possible "
                            "into plain text. Preserve the original wording. "
                            "Do not summarize or add commentary; just output the diary content."
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


def ocr_pdf_with_openai(pdf_path: str) -> str:
    """
    Convert a multipage PDF to text using OpenAI vision, one page at a time.
    """
    images = pdf_to_images(pdf_path)
    all_text = []

    for i, img in enumerate(images):
        print(f"Processing page {i + 1}/{len(images)} with OpenAI Vision...")
        page_text = ocr_page_with_openai(img)
        all_text.append(f"=== PAGE {i + 1} ===\n{page_text.strip()}\n")

    return "\n\n".join(all_text)


def main():
    if len(sys.argv) < 2:
        print("Usage: python ocr_cloud.py input/YourFile.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print("File not found:", pdf_path)
        sys.exit(1)

    print(f"Running cloud OCR on: {pdf_path}")
    text = ocr_pdf_with_openai(pdf_path)

    # Save to output/
    os.makedirs("output", exist_ok=True)
    base_name = os.path.basename(pdf_path)
    name_without_ext = os.path.splitext(base_name)[0]
    out_path = os.path.join("output", f"{name_without_ext}_openai.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    print("\nDone.")
    print(f"Output saved to: {out_path}")


if __name__ == "__main__":
    main()
