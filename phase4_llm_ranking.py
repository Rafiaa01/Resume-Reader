"""
PHASE 4: AI-Powered Resume Ranking
Tries sentence-transformers first, falls back to enhanced TF-IDF if unavailable.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import re, time

# ── JOB DESCRIPTION — Change this ────────────────────────────────────
JOB_DESCRIPTION = """
We are looking for a Python Developer with experience in machine learning,
data analysis, and API development. Required skills include Python, pandas,
scikit-learn, SQL, REST APIs, and Git. Experience with deep learning or NLP
is a plus. Minimum 2 years of experience required. Strong communication and
problem-solving skills needed.
"""

# ── CONFIG ────────────────────────────────────────────────────────────
INPUT_CSV  = r'C:\Users\PMYLS\Desktop\rafiapython\resume_reader\resumes_with_skills.csv'
OUTPUT_CSV = r'C:\Users\PMYLS\Desktop\rafiapython\resume_reader\ranked_resumes_ai.csv'
TOP_N      = 20
# ─────────────────────────────────────────────────────────────────────

REQUIRED_SKILLS = ['python', 'sql', 'git', 'pandas', 'scikit-learn', 'rest api', 'api']
NICE_TO_HAVE    = ['deep learning', 'nlp', 'tensorflow', 'pytorch', 'docker', 'aws']


def clean_text(text, max_chars=1000):
    text = str(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:max_chars]


def skill_score(text):
    text = str(text).lower()
    req  = sum(1 for s in REQUIRED_SKILLS if s in text)
    nice = sum(1 for s in NICE_TO_HAVE   if s in text)
    return round((req/len(REQUIRED_SKILLS))*70 + (nice/len(NICE_TO_HAVE))*30, 2)


def experience_score(years):
    try:
        y = int(years)
        return min(y * 2, 10)
    except Exception:
        return 0


# ── Try sentence-transformers (AI model) ─────────────────────────────
def try_sentence_transformers(df, job_desc):
    try:
        from sentence_transformers import SentenceTransformer, util
        import torch
        print("AI model loaded: sentence-transformers (all-MiniLM-L6-v2)")
        print("First run downloads ~90MB model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        texts = df['clean'].tolist()
        job_emb     = model.encode(clean_text(job_desc), convert_to_tensor=True)
        resume_embs = model.encode(texts, batch_size=64,
                                   convert_to_tensor=True, show_progress_bar=True)
        scores = util.cos_sim(job_emb, resume_embs)[0].cpu().numpy()
        return (scores * 100).round(2), 'sentence-transformers'
    except Exception as e:
        print(f"sentence-transformers unavailable ({e.__class__.__name__})")
        return None, None


# ── Try fastembed (ONNX-based, no PyTorch) ───────────────────────────
def try_fastembed(df, job_desc):
    try:
        from fastembed import TextEmbedding
        import numpy as np
        print("AI model loaded: fastembed (BAAI/bge-small-en-v1.5)")
        model   = TextEmbedding('BAAI/bge-small-en-v1.5')
        texts   = df['clean'].tolist()
        job_emb = list(model.embed([clean_text(job_desc)]))[0]
        print("Encoding resumes...")
        res_embs = np.array(list(model.embed(texts)))
        job_emb  = np.array(job_emb)
        scores   = (res_embs @ job_emb) / (
            np.linalg.norm(res_embs, axis=1) * np.linalg.norm(job_emb) + 1e-10
        )
        return (scores * 100).round(2), 'fastembed'
    except Exception as e:
        print(f"fastembed unavailable ({e.__class__.__name__})")
        return None, None


# ── Fallback: Enhanced TF-IDF ─────────────────────────────────────────
def tfidf_score(df, job_desc):
    print("Using enhanced TF-IDF (no AI model needed)")
    texts = [clean_text(job_desc)] + df['clean'].tolist()
    vec   = TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english')
    mat   = vec.fit_transform(texts)
    sims  = cosine_similarity(mat[0:1], mat[1:])[0]
    return (sims * 100).round(2), 'tfidf'


def run():
    start = time.time()

    print("Loading resumes...")
    df = pd.read_csv(INPUT_CSV)
    df['text']  = df['text'].fillna('').astype(str)
    df['clean'] = df['text'].apply(clean_text)
    print(f"Loaded {len(df)} resumes.")

    # Try AI models in order, fall back if unavailable
    scores, method = try_sentence_transformers(df, JOB_DESCRIPTION)
    if scores is None:
        scores, method = try_fastembed(df, JOB_DESCRIPTION)
    if scores is None:
        scores, method = tfidf_score(df, JOB_DESCRIPTION)

    print(f"\nScoring method used: {method}")

    df['semantic_score']   = scores
    df['skill_score']      = df['text'].apply(skill_score)
    df['experience_score'] = df.get('experience_years',
                             pd.Series([0]*len(df))).apply(experience_score)

    # Final weighted score
    df['final_score'] = (
        df['semantic_score']   * 0.50 +
        df['skill_score']      * 0.35 +
        df['experience_score'] * 1.50
    ).round(2)

    df = df.sort_values('final_score', ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = 'rank'

    # Save
    out_cols = ['file_name','category','name','email','experience_years',
                'skills_flat','semantic_score','skill_score',
                'experience_score','final_score']
    out_cols = [c for c in out_cols if c in df.columns]
    df[out_cols].to_csv(OUTPUT_CSV, index=True, encoding='utf-8-sig')

    elapsed = round((time.time()-start)/60, 1)

    print(f"\n{'='*65}")
    print(f"  RANKING COMPLETE  |  Method: {method}  |  Time: {elapsed} min")
    print(f"{'='*65}")
    print(f"\nTOP {TOP_N} CANDIDATES:\n")
    disp = [c for c in ['file_name','category','name','experience_years',
                         'semantic_score','skill_score','final_score'] if c in df.columns]
    print(df[disp].head(TOP_N).to_string())

    print(f"\nScore stats:")
    print(f"  Top    : {df['final_score'].max():.2f}")
    print(f"  Avg    : {df['final_score'].mean():.2f}")
    print(f"  Bottom : {df['final_score'].min():.2f}")

    if 'category' in df.columns:
        print(f"\nBest candidate per category (top 15):")
        best = (df.groupby('category')
                  .first()[['file_name','final_score']]
                  .sort_values('final_score', ascending=False))
        print(best.head(15).to_string())

    print(f"\nFull results saved to: {OUTPUT_CSV}")
    return df


if __name__ == '__main__':
    df = run()
