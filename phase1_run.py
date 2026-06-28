"""
PHASE 1: Extract text from all 8,905 resume PDFs using OCR
Saves results to extracted_resumes.csv
"""

import os, sys
import pdfplumber
import pytesseract
import pandas as pd
from pdf2image import convert_from_path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\PMYLS\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
POPPLER_PATH  = r'C:\poppler\poppler-24.08.0\Library\bin'
RESUME_ROOT   = r'C:\Users\PMYLS\Downloads\archive\Resumes PDF'
OUTPUT_CSV    = r'C:\Users\PMYLS\Desktop\rafiapython\resume_reader\extracted_resumes.csv'
CHECKPOINT    = r'C:\Users\PMYLS\Desktop\rafiapython\resume_reader\checkpoint.csv'
# ─────────────────────────────────────────────────────────────────────


def extract_text_pdfplumber(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = '\n'.join(p.extract_text() or '' for p in pdf.pages)
        return text.strip()
    except Exception:
        return ''


def extract_text_ocr(pdf_path):
    try:
        images = convert_from_path(
            pdf_path, dpi=150,
            poppler_path=POPPLER_PATH,
        )
        return '\n'.join(pytesseract.image_to_string(img, lang='eng') for img in images).strip()
    except Exception as e:
        return ''


def extract_text(pdf_path):
    text = extract_text_pdfplumber(pdf_path)
    if len(text) > 50:
        return text, 'pdfplumber'
    text = extract_text_ocr(pdf_path)
    return text, 'ocr'


def collect_pdfs(root):
    paths = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith('.pdf'):
                paths.append(os.path.join(dirpath, f))
    return paths


def load_checkpoint():
    """Resume from where we left off if interrupted."""
    if os.path.exists(CHECKPOINT):
        df = pd.read_csv(CHECKPOINT)
        done = set(df['file_path'].tolist())
        print(f"Resuming from checkpoint — {len(done)} already done.")
        return df.to_dict('records'), done
    return [], set()


def run():
    all_pdfs = collect_pdfs(RESUME_ROOT)
    print(f"Total PDFs found: {len(all_pdfs)}")

    results, done_paths = load_checkpoint()
    remaining = [p for p in all_pdfs if p not in done_paths]
    print(f"Remaining to process: {len(remaining)}")

    batch_size = 100   # save checkpoint every 100 files

    for i, pdf_path in enumerate(tqdm(remaining, desc="Extracting")):
        text, method = extract_text(pdf_path)
        results.append({
            'file_path'  : pdf_path,
            'file_name'  : os.path.basename(pdf_path),
            'category'   : os.path.basename(os.path.dirname(pdf_path)),
            'text'       : text,
            'method'     : method,
            'char_count' : len(text),
        })

        # Save checkpoint every 100 files
        if (i + 1) % batch_size == 0:
            pd.DataFrame(results).to_csv(CHECKPOINT, index=False, encoding='utf-8-sig')
            tqdm.write(f"  ✅ Checkpoint saved ({len(results)} done)")

    # Final save
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    # Also save final checkpoint
    df.to_csv(CHECKPOINT, index=False, encoding='utf-8-sig')

    empty = df[df['char_count'] < 10]
    print(f"\n{'='*50}")
    print(f"✅ PHASE 1 COMPLETE")
    print(f"   Total processed : {len(df)}")
    print(f"   Extracted OK    : {len(df) - len(empty)}")
    print(f"   Empty/Failed    : {len(empty)}")
    print(f"   Saved to        : {OUTPUT_CSV}")
    print(f"{'='*50}")

    print("\nSample result:")
    print(df[['file_name','category','method','char_count']].head(5).to_string(index=False))

    return df


if __name__ == '__main__':
    df = run()
