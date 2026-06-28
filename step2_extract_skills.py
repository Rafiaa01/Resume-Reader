"""
STEP 2: Extract Skills, Education, and Experience from resume text
"""

import pandas as pd
import re

# ── SKILL KEYWORDS (expand this list as needed) ──────────────────────
SKILLS_DB = {
    'Programming':   ['python', 'java', 'javascript', 'c++', 'c#', 'r', 'sql', 'php', 'swift', 'kotlin', 'go', 'rust'],
    'Web':           ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'fastapi', 'spring'],
    'Data Science':  ['machine learning', 'deep learning', 'nlp', 'pandas', 'numpy', 'tensorflow', 'pytorch', 'scikit-learn', 'keras', 'opencv'],
    'Databases':     ['mysql', 'postgresql', 'mongodb', 'sqlite', 'oracle', 'redis', 'firebase'],
    'Cloud':         ['aws', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes', 'ci/cd'],
    'Tools':         ['git', 'github', 'jira', 'linux', 'excel', 'power bi', 'tableau'],
    'Soft Skills':   ['communication', 'teamwork', 'leadership', 'problem solving', 'time management'],
}

# ── EDUCATION KEYWORDS ───────────────────────────────────────────────
EDUCATION_KEYWORDS = [
    'bachelor', 'master', 'phd', 'b.sc', 'm.sc', 'b.e', 'm.e', 'b.tech', 'm.tech',
    'mba', 'bba', 'diploma', 'associate', 'degree', 'university', 'college', 'institute',
    'computer science', 'engineering', 'information technology', 'software',
]

# ── EXPERIENCE PATTERNS ──────────────────────────────────────────────
EXPERIENCE_PATTERNS = [
    r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
    r'experience\s*(?:of\s*)?(\d+)\+?\s*years?',
    r'(\d+)\s*-\s*(\d+)\s*years?',
]


def extract_skills(text):
    """Find all matching skills in the text."""
    text_lower = text.lower()
    found = {}
    for category, skills in SKILLS_DB.items():
        matched = [s for s in skills if s in text_lower]
        if matched:
            found[category] = matched
    return found


def extract_education(text):
    """Find education-related sentences."""
    text_lower = text.lower()
    lines = text.split('\n')
    edu_lines = []
    for line in lines:
        if any(kw in line.lower() for kw in EDUCATION_KEYWORDS):
            cleaned = line.strip()
            if len(cleaned) > 10:
                edu_lines.append(cleaned)
    return ' | '.join(edu_lines[:5])       # return top 5 education lines


def extract_experience_years(text):
    """Extract number of years of experience mentioned."""
    text_lower = text.lower()
    for pattern in EXPERIENCE_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            return int(match.group(1))
    return 0


def extract_email(text):
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return match.group(0) if match else ''


def extract_phone(text):
    match = re.search(r'[\+\(]?[1-9][0-9 \-\(\)]{7,14}[0-9]', text)
    return match.group(0) if match else ''


def extract_name(text):
    """Rough name extraction — first non-empty line is usually the name."""
    for line in text.split('\n'):
        line = line.strip()
        if len(line) > 3 and len(line) < 50 and line.replace(' ', '').isalpha():
            return line
    return ''


def process_skills_from_csv(input_csv='extracted_resumes.csv',
                             output_csv='resumes_with_skills.csv'):
    df = pd.read_csv(input_csv)
    df['text'] = df['text'].fillna('')

    print(f"Processing {len(df)} resumes for skill extraction...")

    df['name']             = df['text'].apply(extract_name)
    df['email']            = df['text'].apply(extract_email)
    df['phone']            = df['text'].apply(extract_phone)
    df['skills']           = df['text'].apply(lambda t: str(extract_skills(t)))
    df['education']        = df['text'].apply(extract_education)
    df['experience_years'] = df['text'].apply(extract_experience_years)

    # Flatten skills into one column
    def all_skills_flat(text):
        found = extract_skills(text)
        all_s = []
        for skills in found.values():
            all_s.extend(skills)
        return ', '.join(all_s)

    df['skills_flat'] = df['text'].apply(all_skills_flat)

    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"Saved to {output_csv}")
    return df


if __name__ == '__main__':
    df = process_skills_from_csv()
    print(df[['file_name', 'name', 'email', 'experience_years', 'skills_flat']].head(10))
