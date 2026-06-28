"""
STEP 3: Compare resumes against a job description and rank candidates
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ── JOB DESCRIPTION (paste your job description here) ────────────────
JOB_DESCRIPTION = """
We are looking for a Python Developer with experience in machine learning,
data analysis, and API development. Required skills include Python, pandas,
scikit-learn, SQL, REST APIs, and Git. Experience with deep learning or NLP
is a plus. Minimum 2 years of experience required.
"""
# ─────────────────────────────────────────────────────────────────────


def clean_text(text):
    """Basic text cleaning."""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def rank_by_tfidf(df, job_description, top_n=20):
    """
    Rank resumes using TF-IDF + Cosine Similarity.
    Simple, fast, works well for keyword matching.
    """
    print("Ranking resumes using TF-IDF cosine similarity...")

    df['clean_text'] = df['text'].apply(clean_text)
    job_clean = clean_text(job_description)

    # Build TF-IDF matrix
    all_texts = [job_clean] + df['clean_text'].tolist()
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),     # unigrams + bigrams
        stop_words='english'
    )
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Cosine similarity between job description and each resume
    job_vector     = tfidf_matrix[0]
    resume_vectors = tfidf_matrix[1:]
    scores = cosine_similarity(job_vector, resume_vectors)[0]

    df['similarity_score'] = scores
    df['rank']             = df['similarity_score'].rank(ascending=False).astype(int)

    df_ranked = df.sort_values('similarity_score', ascending=False)
    return df_ranked


def score_by_skills(df, required_skills, nice_to_have=None):
    """
    Add a bonus score based on required vs nice-to-have skills.
    """
    required_skills  = [s.lower() for s in required_skills]
    nice_to_have     = [s.lower() for s in (nice_to_have or [])]

    def skill_score(text):
        text = str(text).lower()
        req_found  = sum(1 for s in required_skills if s in text)
        nice_found = sum(1 for s in nice_to_have  if s in text)
        req_score  = (req_found  / len(required_skills))  * 70   # 70% weight
        nice_score = (nice_found / max(len(nice_to_have), 1)) * 30  # 30% weight
        return round(req_score + nice_score, 2)

    df['skill_score'] = df['text'].apply(skill_score)
    return df


def final_ranking(df, job_description, top_n=20):
    """Combine TF-IDF similarity + skill score for final ranking."""

    # Required & nice-to-have skills from job description (customize)
    required_skills  = ['python', 'sql', 'git', 'pandas', 'scikit-learn', 'rest api']
    nice_to_have     = ['deep learning', 'nlp', 'tensorflow', 'pytorch', 'docker']

    df = rank_by_tfidf(df, job_description, top_n)
    df = score_by_skills(df, required_skills, nice_to_have)

    # Final score = 60% TF-IDF + 40% skill score
    df['final_score'] = (df['similarity_score'] * 60) + (df['skill_score'] * 0.4)
    df['final_score'] = df['final_score'].round(4)

    df = df.sort_values('final_score', ascending=False).reset_index(drop=True)
    df.index += 1   # rank starts at 1

    return df


def display_top_candidates(df, top_n=10):
    cols = ['file_name', 'name', 'email', 'experience_years',
            'skills_flat', 'similarity_score', 'skill_score', 'final_score']
    cols = [c for c in cols if c in df.columns]
    print(f"\nTOP {top_n} CANDIDATES:")
    print(df[cols].head(top_n).to_string(index=True))


if __name__ == '__main__':
    # Load the CSV from Step 2
    try:
        df = pd.read_csv('resumes_with_skills.csv')
    except FileNotFoundError:
        # If Step 2 not done yet, load from Step 1
        df = pd.read_csv('extracted_resumes.csv')
        df['name'] = df['file_name']
        df['email'] = ''
        df['experience_years'] = 0
        df['skills_flat'] = ''

    df['text'] = df['text'].fillna('')

    # Rank resumes
    df_ranked = final_ranking(df, JOB_DESCRIPTION)

    # Show top 10
    display_top_candidates(df_ranked, top_n=10)

    # Save ranked results
    df_ranked.to_csv('ranked_resumes.csv', index=True, encoding='utf-8-sig')
    print("\nRanked results saved to ranked_resumes.csv")
