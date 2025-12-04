import os
import sys
import io
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
try:
    import pytesseract
    HAS_TESSERACT = True
    # Auto-detect Tesseract location (common on macOS with Homebrew)
    import shutil
    tesseract_path = shutil.which('tesseract')
    if not tesseract_path:
        # Check common Homebrew locations
        for path in ['/opt/homebrew/bin/tesseract', '/usr/local/bin/tesseract']:
            if os.path.exists(path):
                tesseract_path = path
                break
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
except ImportError:
    HAS_TESSERACT = False
try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False

def pdf_to_image(pdf_path: str, page_num: int = 0, dpi: int = 300) -> Image.Image:
    """
    Converts a PDF page to a PIL Image object with high resolution.
    """
    if not HAS_PYMUPDF:
        raise RuntimeError("PyMuPDF (fitz) is required to process PDF files. Install it with: pip install PyMuPDF")
    
    doc = fitz.open(pdf_path)
    if page_num >= len(doc):
        raise ValueError(f"Page {page_num} not found. PDF has {len(doc)} pages.")
    
    page = doc[page_num]
    # Higher resolution for better OCR (300 DPI equivalent)
    zoom = dpi / 72.0  # 72 is default DPI
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    # Convert to PIL Image
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img

def preprocess_image(img: Image.Image) -> Image.Image:
    """
    Preprocesses image to improve OCR accuracy:
    - Increase resolution if needed
    - Convert to grayscale
    - Enhance contrast (moderate)
    - Apply light denoising
    - Sharpen slightly
    """
    # Ensure minimum resolution (at least 300 DPI equivalent)
    min_dpi = 300
    current_dpi = 72  # Default assumption
    if hasattr(img, 'info') and 'dpi' in img.info:
        current_dpi = img.info['dpi'][0] if isinstance(img.info['dpi'], tuple) else img.info['dpi']
    
    # Resize if resolution is too low
    if current_dpi < min_dpi:
        scale = min_dpi / current_dpi
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    # Convert to grayscale for better OCR
    if img.mode != 'L':
        img = img.convert('L')
    
    # Enhance contrast moderately (too much can hurt)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)  # Moderate contrast increase
    
    # Light denoising
    img = img.filter(ImageFilter.MedianFilter(size=3))
    
    # Enhance sharpness slightly
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.2)
    
    # Convert back to RGB (some OCR engines prefer RGB)
    img = img.convert('RGB')
    
    return img

def load_image(image_path: str, preprocess: bool = True) -> Image.Image:
    """
    Loads an image file or PDF (first page) and returns a PIL Image object.
    Optionally preprocesses the image for better OCR.
    """
    ext = image_path.lower().split(".")[-1]
    
    if ext == "pdf":
        img = pdf_to_image(image_path, page_num=0, dpi=300)
    else:
        img = Image.open(image_path)
    
    if preprocess:
        img = preprocess_image(img)
    
    return img

# Global EasyOCR reader (initialized once)
_easyocr_reader = None

def ocr_with_easyocr(img: Image.Image) -> str:
    """
    Performs OCR using EasyOCR (better for handwriting).
    """
    global _easyocr_reader
    
    if not HAS_EASYOCR:
        return None
    
    try:
        # Initialize EasyOCR reader once (cached)
        if _easyocr_reader is None:
            _easyocr_reader = easyocr.Reader(['en'], gpu=False)
        
        # Convert PIL Image to numpy array
        img_array = np.array(img)
        
        # Perform OCR
        results = _easyocr_reader.readtext(img_array)
        
        # Combine all detected text
        text_lines = []
        for (bbox, text, confidence) in results:
            if confidence > 0.3:  # Filter low confidence results
                text_lines.append(text)
        
        return '\n'.join(text_lines)
    except Exception as e:
        print(f"EasyOCR error: {e}")
        return None

def ocr_with_tesseract(img: Image.Image, psm_mode: int = 6) -> str:
    """
    Performs OCR using Tesseract with specified PSM mode.
    """
    if not HAS_TESSERACT:
        return None
    
    # Try different PSM modes for better results
    # PSM 6: Assume uniform block of text
    # PSM 11: Sparse text (handwriting often works better)
    # PSM 12: Single line
    # PSM 13: Raw line (treat image as single text line)
    # Don't use whitelist for handwriting as it may exclude valid characters
    config = f'--psm {psm_mode}'
    text = pytesseract.image_to_string(img, config=config)
    return text.strip()

def ocr_image(image_path: str, use_easyocr: bool = True) -> str:
    """
    Performs OCR on the image and returns the extracted text.
    Tries EasyOCR first (better for handwriting), falls back to Tesseract.
    """
    # Load and preprocess the image
    img = load_image(image_path, preprocess=True)
    
    # Try EasyOCR first (much better for handwriting)
    if use_easyocr and HAS_EASYOCR:
        print("Attempting EasyOCR (better for handwriting)...")
        try:
            text = ocr_with_easyocr(img)
            if text and len(text.strip()) > 10:  # If we got reasonable results
                print("EasyOCR succeeded!")
                return text.strip()
        except Exception as e:
            print(f"EasyOCR failed: {e}")
        print("Falling back to Tesseract...")
    
    # Fall back to Tesseract with multiple PSM modes
    if HAS_TESSERACT:
        print("Using Tesseract OCR with improved settings...")
        best_text = ""
        best_length = 0
        
        # Try different PSM modes and keep the best result
        for psm in [11, 6, 12, 3, 13]:  # 11 is often best for handwriting, 13 for raw line
            print(f"  Trying PSM mode {psm}...")
            text = ocr_with_tesseract(img, psm_mode=psm)
            if text and len(text.strip()) > best_length:
                best_text = text.strip()
                best_length = len(best_text)
        
        if best_text:
            print(f"Best result from PSM mode (length: {best_length} chars)")
            return best_text
        else:
            return best_text if best_text else ""
    
    # If neither is available
    raise RuntimeError(
        "No OCR engine available. Install one of:\n"
        "  EasyOCR (recommended for handwriting): pip install easyocr\n"
        "  Tesseract: pip install pytesseract && brew install tesseract"
    )

def main():
    if len(sys.argv) < 2:
        print("Usage: python ocr_one_page.py path/to/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        sys.exit(1)

    print(f"Running OCR on: {image_path} ...")
    text = ocr_image(image_path)

    print("\n===== OCR RESULT (preview) =====\n")
    print(text)
    print("\n===== END PREVIEW =====\n")

    # Save to output/ with same basename but .txt
    base_name = os.path.basename(image_path)
    name_without_ext = os.path.splitext(base_name)[0]
    os.makedirs("output", exist_ok=True)
    out_path = os.path.join("output", f"{name_without_ext}.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Full text saved to: {out_path}")

if __name__ == "__main__":
    main()
