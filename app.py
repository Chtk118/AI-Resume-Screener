import streamlit as st
import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util
from fuzzywuzzy import fuzz

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Helper functions
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def match_skills(resume_text, skills):
    matched, missing = [], []
    for skill in skills:
        if fuzz.partial_ratio(skill.lower(), resume_text.lower()) > 70:
            matched.append(skill)
        else:
            missing.append(skill)
    return matched, missing

# Streamlit UI
st.title("AI-Powered Resume Screening System")
uploaded_file = st.file_uploader("Upload Resume Dataset (CSV)", type="csv")
job_description = st.text_area("Enter Job Description")

if uploaded_file and job_description:
    df = pd.read_csv(uploaded_file)
    df['Resume_Text'] = (df['Skills'].fillna('') + ' ' + df['Current_Job_Title'].fillna('') + ' ' +
                         df['Previous_Job_Titles'].fillna('') + ' ' + df['Education_Level'].fillna('') + ' ' +
                         df['Degrees'].fillna(''))
    df['Cleaned_Resume'] = df['Resume_Text'].apply(clean_text)
    cleaned_job_desc = clean_text(job_description)
    
    job_emb = model.encode(cleaned_job_desc, convert_to_tensor=True)
    resume_embs = model.encode(df['Cleaned_Resume'].tolist(), convert_to_tensor=True)
    scores = util.cos_sim(resume_embs, job_emb).cpu().numpy().flatten()
    df['Match_Score'] = scores

    job_skills = re.findall(r'\b\w+\b', cleaned_job_desc)[:20]
    df['Matched_Skills'] = df['Cleaned_Resume'].apply(lambda x: match_skills(x, job_skills)[0])
    df['Missing_Skills'] = df['Cleaned_Resume'].apply(lambda x: match_skills(x, job_skills)[1])

    df_sorted = df.sort_values(by='Match_Score', ascending=False)
    st.subheader("Top Matching Resumes")
    st.dataframe(df_sorted[['Name', 'Match_Score', 'Matched_Skills', 'Missing_Skills']].head(10))
