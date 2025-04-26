import streamlit as st
import google.generativeai as genai
import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input_text):
    model = genai.GenerativeModel(model_name="models/gemini-1.5-pro")
    response = model.generate_content(input_text)
    return response.text

def input_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    return "".join([page.extract_text() for page in reader.pages])

st.title("🎯 Smart ATS Resume Checker")
st.text("📈 Improve your Resume with AI")

jd = st.text_area("📄 Paste the Job Description here")
uploaded_file = st.file_uploader("📎 Upload your Resume (PDF only)", type="pdf")
submit = st.button("✅ Submit")

if submit and uploaded_file and jd.strip():
    with st.spinner("Analyzing resume..."):
        resume_text = input_pdf_text(uploaded_file)

        final_input = f"""
You are an expert ATS (Application Tracking System) evaluator specialized in the tech field.

Strictly analyze the given Resume against the provided Job Description. 
The job market is highly competitive, so provide precise feedback.

Please give your response in the following simple format, without any extra explanation:

JD Match: (percentage)%
Missing Keywords: (comma-separated list of keywords)
Profile Summary: (1-2 lines summary about the resume quality)

Resume:
{resume_text}

Job Description:
{jd}
"""
        response = get_gemini_response(final_input)

        if response:
            st.text_area("📋 ATS Resume Report", response, height=400)
        else:
            st.warning("No response received. Check your API settings.")
else:
    st.warning("Please upload Resume and Job Description to continue.")
