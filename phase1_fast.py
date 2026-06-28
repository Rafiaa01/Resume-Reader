"""
PHASE 1 (FAST): Extract text from 8,905 PDFs using multiprocessing
Runs 4 workers in parallel — ~4x faster than single-threaded.
Auto-saves checkpoint every 200 files so you can resume if interrupted.
"""

import os, sys, time
import pdfplumber
import pytesseract
import pandas as pd
from pdf2image import convert_from_path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────
TESSERACT    = r'C:\Users\PMYLS\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
POPPLER      = r'C:\poppler\poppler-24.08.0\Library\bin'
RESUME_ROOT  = r'C:\Users\PMYLS\Downloads\archive\Resumes PDF'
OUTPUT_CSV   = r'C:\Users\PMYLS\Desktop\rafiapython\resume_reader\extracted_resumes.csv'
CHECKPOINT   = r'C:\Users\PMYLS\Desktop\rafiapython\resume_reader\checkpoint.csv'
WORKERS      = 8       # using 8 of 12 cores
BATCH_SIZE   = 200     # save checkpoint every N files
# ─────────────────────────────────────────────────────────────────────


def init_worker():
    """Each worker sets its own Tesseract path."""
    pytesseract.pytesseract.tesseract_cmd = TESSERACT


def extract_one(pdf_path):
    """Extract text from a single PDF — called by each worker."""
    file_name = os.path.basename(pdf_path)
    category  = os.path.basename(os.path.dirname(pdf_path))

    # Try pdfplumber first (fast for text PDFs)
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = '\n'.join(p.extract_text() or '' for p in pdf.pages).strip()
        if len(text) > 50:
            return {
                'file_path' : pdf_path,
                'file_name' : file_name,
                'category'  : category,
                'text'      : text,
                'method'    : 'pdfplumber',
                'char_count': len(text),
            }
    except Exception:
        pass

    # Fall back to OCR
    try:
        images = convert_from_path(
            pdf_path, dpi=100,          # lower DPI = faster, still readable
            poppler_path=POPPLER,
            first_page=1, last_page=3,  # max 3 pages (most resumes are 1-2)
        )
        # --oem 1 = LSTM engine only (faster), --psm 3 = auto page segmentation
        config = '--oem 1 --psm 3'
        text = '\n'.join(
            pytesseract.image_to_string(img, lang='eng', config=config)
            for img in images
        ).strip()
    except Exception as e:
        text = ''

    return {
        'file_path' : pdf_path,
        'file_name' : file_name,
        'category'  : category,
        'text'      : text,
        'method'    : 'ocr',
        'char_count': len(text),
    }


def collect_pdfs(root):
    paths = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith('.pdf'):
                paths.append(os.path.join(dirpath, f))
    return sorted(paths)


def load_checkpoint():
    if os.path.exists(CHECKPOINT):
        df = pd.read_csv(CHECKPOINT)
        done = set(df['file_path'].tolist())
        print(f"Resuming checkpoint — {len(done)} already processed.")
        return df.to_dict('records'), done
    return [], set()


def run():
    start = time.time()
    all_pdfs = collect_pdfs(RESUME_ROOT)
    print(f"Total PDFs : {len(all_pdfs)}")
    print(f"Workers    : {WORKERS}")

    results, done = load_checkpoint()
    remaining = [p for p in all_pdfs if p not in done]
    print(f"Remaining  : {len(remaining)}")

    if not remaining:
        print("Nothing to do — all PDFs already extracted!")
        return pd.read_csv(CHECKPOINT)

    batch = []
    with Pool(processes=WORKERS, initializer=init_worker) as pool:
        for result in tqdm(
            pool.imap_unordered(extract_one, remaining, chunksize=4),
            total=len(remaining),
            desc="Extracting",
            unit="pdf"
        ):
            results.append(result)
            batch.append(result)

            # Save checkpoint every BATCH_SIZE files
            if len(batch) >= BATCH_SIZE:
                pd.DataFrame(results).to_csv(CHECKPOINT, index=False, encoding='utf-8-sig')
                batch = []
                elapsed = time.time() - start
                done_count = len(results)
                remaining_count = len(remaining) - done_count
                eta_sec = (elapsed / done_count) * remaining_count if done_count else 0
                eta_min = int(eta_sec / 60)
                tqdm.write(f"  >> {done_count} done | ETA: ~{eta_min} min")

    # Final save
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    df.to_csv(CHECKPOINT, index=False, encoding='utf-8-sig')

    elapsed_min = int((time.time() - start) / 60)
    empty_count = len(df[df['char_count'] < 10])

    print(f"\n{'='*55}")
    print(f"  PHASE 1 COMPLETE")
    print(f"  Total processed : {len(df)}")
    print(f"  Extracted OK    : {len(df) - empty_count}")
    print(f"  Empty/Failed    : {empty_count}")
    print(f"  Time taken      : {elapsed_min} minutes")
    print(f"  Saved to        : {OUTPUT_CSV}")
    print(f"{'='*55}")

    # Category breakdown
    print("\nResumes by category:")
    print(df.groupby('category')['char_count'].agg(['count','mean']).round(0).to_string())

    return df


if __name__ == '__main__':
    df = run()
