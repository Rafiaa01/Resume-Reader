"""
STEP 1: Extract text from all PDF resumes (handles both text-based and scanned/image PDFs)
"""

import os
import pdfplumber
import pytesseract
import pandas as pd
from pdf2image import convert_from_path
from tqdm import tqdm          # progress bar
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ──────────────────────────────────────────────────────────
# Tesseract path (already installed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\PMYLS\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# Poppler path (download and extract to C:\poppler)
POPPLER_PATH = r'C:\poppler\Library\bin'

# Folder that contains ALL your resume PDF subfolders
RESUME_ROOT = r'C:\path\to\your\resumes'   # <-- CHANGE THIS to your folder

# Output CSV file
OUTPUT_CSV = 'extracted_resumes.csv'
# ────────────────────────────────────────────────────────────────────


def extract_text_pdfplumber(pdf_path):
    """Try to extract text directly (works for text-based PDFs)."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ''
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + '\n'
        return text.strip()
    except Exception:
        return ''


def extract_text_ocr(pdf_path):
    """Use OCR for scanned/image-based PDFs."""
    try:
        images = convert_from_path(
            pdf_path,
            dpi=200,                      # 200 dpi = good balance of speed vs accuracy
            poppler_path=POPPLER_PATH
        )
        text = ''
        for img in images:
            t = pytesseract.image_to_string(img, lang='eng')
            text += t + '\n'
        return text.strip()
    except Exception as e:
        return f'OCR_ERROR: {e}'


def extract_text(pdf_path):
    """
    Smart extraction:
    1. Try pdfplumber (fast, accurate for text PDFs)
    2. If empty → fall back to OCR (for scanned PDFs)
    """
    text = extract_text_pdfplumber(pdf_path)
    if len(text) > 50:          # if we got meaningful text, use it
        return text, 'pdfplumber'
    else:
        text = extract_text_ocr(pdf_path)
        return text, 'ocr'


def collect_pdf_paths(root_folder):
    """Walk through all subfolders and collect PDF paths."""
    pdf_paths = []
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith('.pdf'):
                pdf_paths.append(os.path.join(dirpath, filename))
    print(f"Found {len(pdf_paths)} PDF files.")
    return pdf_paths


def process_all_resumes(root_folder, output_csv):
    pdf_paths = collect_pdf_paths(root_folder)

    results = []
    failed  = []

    for pdf_path in tqdm(pdf_paths, desc="Extracting resumes"):
        text, method = extract_text(pdf_path)

        if 'OCR_ERROR' in text or len(text.strip()) < 10:
            failed.append(pdf_path)
            text = ''

        results.append({
            'file_path':    pdf_path,
            'file_name':    os.path.basename(pdf_path),
            'folder':       os.path.basename(os.path.dirname(pdf_path)),
            'text':         text,
            'method':       method,
            'text_length':  len(text),
        })

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    print(f"\n✅ Done!")
    print(f"   Total PDFs    : {len(pdf_paths)}")
    print(f"   Extracted OK  : {len(pdf_paths) - len(failed)}")
    print(f"   Failed/Empty  : {len(failed)}")
    print(f"   Saved to      : {output_csv}")

    if failed:
        print("\n⚠️  Failed files:")
        for f in failed[:10]:
            print(f"   {f}")

    return df


# ── TEST SINGLE FILE FIRST ───────────────────────────────────────────
def test_single_pdf(pdf_path):
    """Test on one PDF before processing all 8905."""
    print(f"Testing: {pdf_path}\n")

    print("--- pdfplumber result ---")
    text = extract_text_pdfplumber(pdf_path)
    print(f"Length: {len(text)} chars")
    print(text[:500] if text else "EMPTY")

    print("\n--- OCR result ---")
    text_ocr = extract_text_ocr(pdf_path)
    print(f"Length: {len(text_ocr)} chars")
    print(text_ocr[:500] if text_ocr else "EMPTY")


# ── RUN ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # FIRST: test on 1 PDF to confirm everything works
    # test_single_pdf(r'C:\path\to\one\resume.pdf')   # <-- uncomment & test first

    # THEN: process all resumes
    df = process_all_resumes(RESUME_ROOT, OUTPUT_CSV)
    print(df.head())
