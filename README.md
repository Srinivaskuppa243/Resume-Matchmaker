# Resume-Job-Matcher

## Overview

This repository contains a Python script for matching resumes with job descriptions based on their content and keywords. The script uses natural language processing techniques, TF-IDF vectorization, and cosine similarity to recommend resumes for specific job categories.

## Requirements

Ensure you have the required Python libraries installed by running:

```bash
pip install -r requirements.txt

```
## Usage

1. Data Preperation:
   Place resumes in the /resumes folder in DOCX or PDF format.
   Ensure job descriptions are in Job_descriptn.csv.
   
3. Run the Script:
   ```bash
    python Resume_recommendation.py
   ```
4. Output:
   View the cosine similarities between job descriptions and resumes.
   See the top 5 recommended resumes for each job category.
   
 ## File structure:
   
   Resumes: Folder containing resumes in DOCX and PDF formats.
   
   Job_descriptn.csv: CSV file with job descriptions.
   
   Resume_recommendation.py: Python script for matching resumes with job descriptions.
   
   requirements.txt: List of required Python packages.

