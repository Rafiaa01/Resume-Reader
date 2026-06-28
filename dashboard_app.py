"""
Resume Screening Dashboard - Flask Backend
Run: python dashboard_app.py
Open: http://127.0.0.1:5000
"""

from flask import Flask, jsonify, request, render_template_string, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import os, re, io
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

CSV_PATH      = r"C:\Users\PMYLS\Desktop\rafiapython\resume_reader\ranked_resumes_ai.csv"
UPLOAD_FOLDER = r"C:\Users\PMYLS\Desktop\rafiapython\resume_reader\uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

_current_jd = {
    "text": (
        "We are looking for a Python Developer with experience in machine learning, "
        "data analysis, and API development. Required skills include Python, pandas, "
        "scikit-learn, SQL, REST APIs, and Git. Experience with deep learning or NLP "
        "is a plus. Minimum 2 years of experience required."
    )
}

ALL_SKILLS = [
    "python","sql","git","pandas","scikit-learn","rest api","api",
    "machine learning","data analysis","flask","django","javascript",
    "react","java","c++","docker","aws","tensorflow","pytorch",
    "nlp","deep learning","html","css","node","mongodb","mysql",
    "flutter","dart","firebase","kotlin","swift","php","ruby",
    "scala","spark","hadoop","tableau","power bi","excel","r",
]


TESSERACT_CMD = r"C:\Users\PMYLS\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
POPPLER_PATH  = r"C:\poppler\poppler-24.08.0\Library\bin"


def extract_text_from_file(file_path, filename):
    ext  = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    text = ""
    try:
        if ext == "pdf":
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages[:5]:
                    t = page.extract_text()
                    if t:
                        text += t + "\n"
            # Scanned PDF — fall back to Tesseract OCR
            if not text.strip():
                try:
                    import pytesseract
                    from pdf2image import convert_from_path
                    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
                    images = convert_from_path(file_path, dpi=150,
                                               poppler_path=POPPLER_PATH, last_page=5)
                    for img in images:
                        text += pytesseract.image_to_string(img, config="--oem 1 --psm 3") + "\n"
                except Exception as ocr_err:
                    text = f"OCR error: {ocr_err}"
        elif ext in ("docx", "doc"):
            from docx import Document
            doc  = Document(file_path)
            text = "\n".join(p.text for p in doc.paragraphs)
        else:
            text = "Unsupported file type."
    except Exception as e:
        text = f"Extraction error: {e}"
    return text.strip()


def score_resume(text, jd_text):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    t_low  = text.lower()
    jd_low = jd_text.lower()

    jd_skills = [s for s in ALL_SKILLS if s in jd_low] or ALL_SKILLS
    matched   = [s for s in jd_skills if s in t_low]
    missing   = [s for s in jd_skills if s not in t_low]
    skill_pct = round(len(matched) / len(jd_skills) * 100, 1) if jd_skills else 0

    clean    = lambda t: re.sub(r"\s+", " ", t)[:2000]
    vec      = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    mat      = vec.fit_transform([clean(jd_text), clean(text)])
    semantic = round(float(cosine_similarity(mat[0:1], mat[1:])[0][0]) * 100, 2)

    exp_match = re.findall(r"(\d+)\s*(?:\+\s*)?years?", t_low)
    exp_years = max((int(x) for x in exp_match if int(x) < 40), default=0)
    exp_score = min(exp_years * 2, 10)
    final     = round(semantic * 0.50 + skill_pct * 0.35 + exp_score * 1.50, 2)

    lines     = [l.strip() for l in text.split("\n") if l.strip()]
    name_guess = lines[0] if lines else "Unknown"

    return {
        "name":             name_guess,
        "final_score":      final,
        "semantic_score":   semantic,
        "skill_score":      skill_pct,
        "experience_years": exp_years,
        "matched_skills":   matched,
        "missing_skills":   missing[:10],
        "recommendation":   "Shortlist" if final >= 40 else "Review" if final >= 20 else "Pass",
    }


def load_data():
    df = pd.read_csv(CSV_PATH)
    df = df.fillna("")
    if "rank" not in df.columns:
        df.insert(0, "rank", range(1, len(df) + 1))
    for col in ["final_score", "semantic_score", "skill_score", "experience_years"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


@app.route("/")
def index():
    return render_template_string(open(
        os.path.join(os.path.dirname(__file__), "dashboard.html"),
        encoding="utf-8"
    ).read())


@app.route("/api/stats")
def stats():
    df = load_data()
    categories = df["category"].value_counts().to_dict() if "category" in df.columns else {}
    return jsonify({
        "total":          int(len(df)),
        "categories":     int(df["category"].nunique()) if "category" in df.columns else 0,
        "avg_score":      round(float(df["final_score"].mean()), 2),
        "top_score":      round(float(df["final_score"].max()), 2),
        "top_categories": dict(list(categories.items())[:8]),
        "current_jd":     _current_jd["text"],
    })


@app.route("/api/candidates")
def candidates():
    df = load_data()
    search    = request.args.get("search",    "").lower()
    category  = request.args.get("category",  "")
    min_exp   = request.args.get("min_exp",   0,  type=int)
    min_score = request.args.get("min_score", 0,  type=float)
    sort_by   = request.args.get("sort",      "final_score")
    page      = request.args.get("page",      1,  type=int)
    per_page  = request.args.get("per_page",  20, type=int)

    if search:
        mask = (
            df["name"].str.lower().str.contains(search, na=False) |
            df["skills_flat"].str.lower().str.contains(search, na=False) |
            df["category"].str.lower().str.contains(search, na=False)
        )
        df = df[mask]
    if category:
        df = df[df["category"].str.lower() == category.lower()]
    if "experience_years" in df.columns:
        df = df[df["experience_years"] >= min_exp]
    df = df[df["final_score"] >= min_score]
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)

    total   = len(df)
    start   = (page - 1) * per_page
    page_df = df.iloc[start:start + per_page]
    return jsonify({
        "total":      total, "page": page, "per_page": per_page,
        "pages":      int(np.ceil(total / per_page)),
        "candidates": page_df.to_dict(orient="records"),
    })


@app.route("/api/categories")
def categories():
    df = load_data()
    if "category" not in df.columns:
        return jsonify([])
    return jsonify(sorted(df["category"].dropna().unique().tolist()))


@app.route("/api/top_by_category")
def top_by_category():
    df = load_data()
    if "category" not in df.columns:
        return jsonify([])
    top = (df.sort_values("final_score", ascending=False)
             .groupby("category").first().reset_index()
             .sort_values("final_score", ascending=False)
             [["category","file_name","name","final_score","experience_years"]]
             .head(15))
    return jsonify(top.to_dict(orient="records"))


@app.route("/api/upload-resume", methods=["POST"])
def upload_resume():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400
    fname = secure_filename(file.filename)
    ext   = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
    if ext not in ("pdf", "docx", "doc"):
        return jsonify({"error": "Only PDF and DOCX files are supported"}), 400
    save_path = os.path.join(UPLOAD_FOLDER, fname)
    file.save(save_path)
    text   = extract_text_from_file(save_path, fname)
    result = score_resume(text, _current_jd["text"])
    result["filename"]     = fname
    result["full_text"]    = text   # kept server-side for add-to-dataset
    result["text_preview"] = text[:300] + "..." if len(text) > 300 else text

    # cache so /api/add-to-dataset can use it without re-extracting
    _upload_cache[fname] = {"text": text, "result": result}
    return jsonify(result)


# ── Add uploaded resume to live dataset ──────────────────────────────────────

_upload_cache = {}   # fname -> {text, result}

@app.route("/api/add-to-dataset", methods=["POST"])
def add_to_dataset():
    data     = request.get_json(silent=True) or {}
    fname    = data.get("filename", "")
    name     = data.get("name", "")
    category = data.get("category", "Uploaded").strip() or "Uploaded"

    # Get cached scored result
    cached = _upload_cache.get(fname)
    if not cached:
        return jsonify({"error": "Resume not found in cache. Please upload again."}), 400

    scored = cached["result"]
    text   = cached["text"]

    # Build new row matching CSV columns
    skills_flat = ", ".join(scored.get("matched_skills", []))
    new_row = {
        "file_name":        fname,
        "category":         category,
        "name":             name or scored.get("name", fname),
        "email":            "",
        "experience_years": scored.get("experience_years", 0),
        "skills_flat":      skills_flat,
        "semantic_score":   scored.get("semantic_score", 0),
        "skill_score":      scored.get("skill_score", 0),
        "experience_score": min(int(scored.get("experience_years", 0)) * 2, 10),
        "final_score":      scored.get("final_score", 0),
        "text":             text[:500],
    }

    # Load CSV, append, re-rank, save
    df      = load_data()
    new_df  = pd.DataFrame([new_row])
    # Only keep columns that exist in the CSV
    for col in df.columns:
        if col not in new_df.columns:
            new_df[col] = ""
    new_df  = new_df[[c for c in df.columns if c in new_df.columns]]

    combined = pd.concat([df, new_df], ignore_index=True)
    combined = combined.sort_values("final_score", ascending=False).reset_index(drop=True)
    combined.index += 1
    combined.index.name = "rank"

    # Find the new entry's rank
    new_rank = int(combined[combined["file_name"] == fname].index[0]) if fname in combined["file_name"].values else "?"

    combined.to_csv(CSV_PATH, index=True, encoding="utf-8-sig")

    # Remove from cache
    _upload_cache.pop(fname, None)

    return jsonify({
        "message":   f"{name or fname} added to dataset and re-ranked",
        "new_rank":  new_rank,
        "new_total": len(combined),
        "final_score": scored.get("final_score", 0),
    })


@app.route("/api/set-jd", methods=["POST"])
def set_jd():
    data = request.get_json(silent=True) or {}
    jd   = (data.get("jd") or "").strip()
    if len(jd) < 20:
        return jsonify({"error": "Job description too short (min 20 chars)"}), 400
    _current_jd["text"] = jd
    return jsonify({"message": "Job description updated", "length": len(jd)})


@app.route("/api/get-jd")
def get_jd():
    return jsonify({"jd": _current_jd["text"]})


@app.route("/api/export/pdf")
def export_pdf():
    try:
        from fpdf import FPDF
    except ImportError:
        return jsonify({"error": "fpdf2 not installed. Run: pip install fpdf2"}), 500

    df  = load_data().head(100)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    from fpdf.enums import XPos, YPos
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Resume Screening Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Top {len(df)} Candidates | AI Resume Screener", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Job Description:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 8)
    pdf.multi_cell(0, 5, _current_jd["text"][:400])
    pdf.ln(4)

    col_w   = [12, 55, 38, 20, 22, 28]
    headers = ["Rank","Name","Category","Exp","Skill %","Score"]
    pdf.set_fill_color(108, 99, 255)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 9)
    for w, h in zip(col_w, headers):
        pdf.cell(w, 8, h, border=1, fill=True)
    pdf.ln()

    pdf.set_font("Helvetica", "", 8)
    for i, (_, row) in enumerate(df.iterrows()):
        fill = i % 2 == 0
        pdf.set_fill_color(240, 240, 255) if fill else pdf.set_fill_color(255, 255, 255)
        pdf.set_text_color(0, 0, 0)
        name = str(row.get("name") or row.get("file_name") or "-")[:30]
        cat  = str(row.get("category") or "-")[:22]
        pdf.cell(col_w[0], 6, str(row.get("rank", i+1)),                  border=1, fill=fill)
        pdf.cell(col_w[1], 6, name,                                        border=1, fill=fill)
        pdf.cell(col_w[2], 6, cat,                                         border=1, fill=fill)
        pdf.cell(col_w[3], 6, str(int(row.get("experience_years", 0))),   border=1, fill=fill, align="C")
        pdf.cell(col_w[4], 6, f"{float(row.get('skill_score', 0)):.0f}%", border=1, fill=fill, align="C")
        pdf.cell(col_w[5], 6, f"{float(row.get('final_score', 0)):.2f}",  border=1, fill=fill, align="C")
        pdf.ln()

    buf = io.BytesIO(pdf.output())
    buf.seek(0)
    return send_file(buf, mimetype="application/pdf",
                     as_attachment=True, download_name="resume_ranking_report.pdf")


@app.route("/api/export/csv")
def export_csv():
    df  = load_data()
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    mem = io.BytesIO(buf.getvalue().encode("utf-8-sig"))
    mem.seek(0)
    return send_file(mem, mimetype="text/csv",
                     as_attachment=True, download_name="resume_ranking.csv")


if __name__ == "__main__":
    print("=" * 55)
    print("  Resume Screening Dashboard")
    print("  Open: http://127.0.0.1:5000")
    print("=" * 55)
    app.run(debug=False, port=5000)
