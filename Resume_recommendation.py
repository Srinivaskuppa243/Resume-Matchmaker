!pip install docx
!pip install PyPDF2
!pip install python-docx
!pip install pdfplumber
!pip install nltk
!pip install torch
#1
import pandas as pd
import glob
import re
import os
from docx import Document
import numpy as np
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from joblib import Parallel, delayed
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
#2
def read_docx(file_path):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)
def read_pdf(file_path):
    pdf_file = open(file_path, 'rb')
    pdf_reader = PyPDF2.PdfFileReader(pdf_file)
    full_text = []
    for page_num in range(pdf_reader.numPages):
        page = pdf_reader.getPage(page_num)
        full_text.append(page.extractText())
    return '\n'.join(full_text)
#3
resume_folder = '/content/sample_data/Resumes'
resume_files = os.listdir(resume_folder)
resume_texts = []
resume_ids=[]
start_id= 101
for i, file  in enumerate(resume_files):
    resume_id=start_id+i
    file_path = os.path.join(resume_folder, file)
    if file.endswith('.docx'):
        text = read_docx(file_path)
    elif file.endswith('.pdf'):
        text = read_pdf(file_path)
    resume_ids.append(resume_id)
    resume_texts.append(text)
df_resume = pd.DataFrame({'resume_id':  resume_ids,'text': resume_texts})
df_resume.drop_duplicates(inplace=True)
def extract_keywords(text):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', use_idf=True, smooth_idf=True, max_features=10)
    tfidf_matrix = tfidf_vectorizer.fit_transform([text])
    feature_names = tfidf_vectorizer.get_feature_names_out()
    keywords = [feature_names[idx] for idx in tfidf_matrix.indices]
    return ", ".join(keywords)  # Join the extracted keywords with commas
#4
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
def clean_resume_text(text):
    cleaned_text = re.sub(r'\n|\t', ' ', text).strip()
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', cleaned_text)
    cleaned_text = cleaned_text.lower()
    cleaned_text = re.sub(r'\S+@\S+', '', cleaned_text)
    cleaned_text = re.sub(r'\d{3} \d{3}\d{4}', '', cleaned_text)
    words = word_tokenize(cleaned_text)
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    cleaned_text = ' '.join(cleaned_words)
    return cleaned_text

df_resume['text'] = df_resume['text'].apply(clean_resume_text)
df_resume['keywords']=df_resume['text'].apply(extract_keywords)
df_resume.drop_duplicates(inplace=True)
df_resume.head(35)
#5
df_job=pd.read_csv('/content/Job_descriptn.csv')
df_job.drop_duplicates(inplace=True)
num_jobs = len(df_job)
job_ids = range(1, num_jobs + 1)
df_job['job_id'] = job_ids
df_job['description'] = df_job['Description'].apply(clean_resume_text)
df_job = df_job[['job_id', 'Job Category', 'description']]
df_job['keywords']=df_job['description'].apply(extract_keywords)
df_job
#6
tfidf_vectorizer = TfidfVectorizer(stop_words='english', use_idf=True, smooth_idf=True)
tfidf_matrix_job = tfidf_vectorizer.fit_transform(df_job['description'])
tfidf_matrix_resume = tfidf_vectorizer.transform(df_resume['text'])
if tfidf_matrix_resume.shape[1] != tfidf_matrix_job.shape[1]:
    raise ValueError("The number of features in the TF-IDF matrices for job descriptions and resumes don't match.")
print("Job Descriptions:")
print(df_job[['job_id', 'Job Category']])
print("\nResume TF-IDF Matrix:")
print(tfidf_matrix_resume.toarray())
print('\nJob TFIDF_Matrix_Job:')
print(tfidf_matrix_job.toarray())

#7
cosine_similarities = np.dot(tfidf_matrix_job, tfidf_matrix_resume.T)
print("\nCosine Similarities:")
print(cosine_similarities)
#8
recommended_resumes = []
for i in range(len(df_job)):
    top_indices = cosine_similarities[i].argsort()[-5:][::-1]  # Get the indices of the top 5 recommendations
    top_resume_ids = df_resume.iloc[top_indices]['resume_id'].tolist()
    job_category = df_job.iloc[i]['Job Category']
    recommended_resumes.append({'Job Category': job_category, 'Recommended Resumes': top_resume_ids})

recommended_resumes_df = pd.DataFrame(recommended_resumes)
print("\nRecommended Resumes:")
print(recommended_resumes_df)