"""
RUN ALL STEPS IN ONE GO
Just set RESUME_ROOT to your folder containing all resume PDFs.
"""

# ── CHANGE THIS ──────────────────────────────────────────────────────
RESUME_ROOT = r'C:\path\to\your\resumes'   # your folder with PDF subfolders

JOB_DESCRIPTION = """
We are looking for a Python Developer with experience in machine learning,
data analysis, and API development. Required skills: Python, pandas,
scikit-learn, SQL, REST APIs, Git. Minimum 2 years of experience.
"""
# ─────────────────────────────────────────────────────────────────────

from step1_extract_text  import process_all_resumes
from step2_extract_skills import process_skills_from_csv
from step3_rank_resumes  import final_ranking, display_top_candidates

print("=" * 60)
print("STEP 1: Extracting text from PDFs...")
print("=" * 60)
df = process_all_resumes(RESUME_ROOT, 'extracted_resumes.csv')

print("\n" + "=" * 60)
print("STEP 2: Extracting skills, education, experience...")
print("=" * 60)
df = process_skills_from_csv('extracted_resumes.csv', 'resumes_with_skills.csv')

print("\n" + "=" * 60)
print("STEP 3: Ranking candidates...")
print("=" * 60)
df_ranked = final_ranking(df, JOB_DESCRIPTION)
display_top_candidates(df_ranked, top_n=20)
df_ranked.to_csv('ranked_resumes.csv', index=True, encoding='utf-8-sig')

print("\n✅ ALL DONE!")
print("   extracted_resumes.csv    → raw text from all PDFs")
print("   resumes_with_skills.csv  → skills + education + experience")
print("   ranked_resumes.csv       → candidates ranked by job match")
