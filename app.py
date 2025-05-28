import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re

st.set_page_config(layout="wide", page_title="Smart ATS Resume Checker")

@st.cache_resource
def load_model():
    try:
        return pipeline("text-classification", model="AventIQ-AI/bert-talentmatchai")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

classifier = load_model()

def input_pdf_text(uploaded_file):
    text = ""
    try:
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            if page_text := page.extract_text():
                text += page_text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None
    return text

def extract_keywords(text):
    words = re.findall(r'\b\w+\b', text.lower())
    keywords = [word for word in words if word not in ENGLISH_STOP_WORDS and len(word) > 2]
    return set(keywords)

PROGRAMMING_LANGUAGES = {
    "python", "java", "c++", "c", "javascript", "typescript", "go", "ruby", "rust", "kotlin", "php", "r", "sql", "scala"
}
STUDY_DOMAINS = {
    "machine learning", "deep learning", "artificial intelligence", "nlp", "cloud computing", 
    "data science", "big data", "computer vision", "cybersecurity", "blockchain", 
    "devops", "database", "networking", "cloud", "analytics"
}

def filter_relevant_keywords(keywords):
    return [word for word in keywords if word in PROGRAMMING_LANGUAGES or word in STUDY_DOMAINS]

# UI
st.title(" Smart ATS Resume Checker")
st.markdown("📈 Improve your Resume with AI")
st.info("💡 Uses 'AventIQ-AI/bert-talentmatchai' to evaluate resume vs job description")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Job Description")
    jd = st.text_area("Paste the Job Description here:", height=400, key="jd_input")

with col2:
    st.subheader("📎 Upload Your Resume")
    uploaded_file = st.file_uploader("Upload your Resume (PDF only)", type="pdf", key="resume_uploader")

st.markdown("---")

submit = st.button("✅ Analyze Resume")

if submit:
    if not jd.strip():
        st.warning("Please paste the **Job Description** to continue.")
    elif not uploaded_file:
        st.warning("Please upload your **Resume (PDF)** to continue.")
    else:
        with st.spinner("Analyzing resume..."):
            resume_text = input_pdf_text(uploaded_file)

            if not resume_text or len(resume_text.strip()) < 50:
                st.error("Not enough resume text extracted. Try another file.")
                st.stop()

            if len(jd.strip()) < 50:
                st.error("Job Description too short. Add more detail.")
                st.stop()

            inputs = {"text": resume_text, "text_pair": jd}
            try:
                result = classifier(inputs, truncation=True)

                if isinstance(result, dict):
                    result = [result]

                if result and 'label' in result[0]:
                    predicted_label = result[0]['label']
                    confidence_score = result[0]['score']

                    label_map = {
                        "LABEL_0": " Not a Good Fit",
                        "LABEL_1": " Potential Fit",
                        "LABEL_2": " Good Fit",
                        "LABEL_3": " Excellent Fit (Experimental)"
                    }

                    match_result = label_map.get(predicted_label, "Unknown Fit Label")

                    st.success("✅ Analysis Complete!")
                    st.markdown("### 📊 ATS Resume Report")
                    st.write(f"**Overall Match:** {match_result}")
                    st.write(f"**Confidence:** {confidence_score:.2%}")

                    # 🔍 Keyword Analysis
                    jd_keywords = extract_keywords(jd)
                    resume_keywords = extract_keywords(resume_text)
                    missing_keywords = jd_keywords - resume_keywords

                    # Only show programming/domain keywords
                    filtered_missing = filter_relevant_keywords(missing_keywords)

                    st.markdown("###  Missing Technical Keywords")
                    if filtered_missing:
                        st.markdown("These important technical/domain keywords are missing in your resume:")
                        for word in sorted(filtered_missing):
                            st.markdown(f"- 🔸 **{word.capitalize()}**")
                    else:
                        st.success("🎉 Awesome! Your resume covers all the essential tech skills and domains!")

                    st.markdown("---")
                    st.markdown("### 💡 Interpretation & Next Steps")

                    if predicted_label == "LABEL_2":
                        st.write("✅ Great fit! Make sure to highlight relevant strengths.")
                    elif predicted_label == "LABEL_1":
                        st.write("⚠️ Potential fit. Add more matching keywords.")
                    elif predicted_label == "LABEL_0":
                        st.write("❌ Low match. Consider revising your resume.")
                    elif predicted_label == "LABEL_3":
                        st.write("🚀 Strong match! (Experimental label detected)")
                    else:
                        st.warning("🤖 Unrecognized label. Model might need updates.")

                    st.markdown("> 📌 Always tailor your resume manually before applying.")
                else:
                    st.error("AI model returned an unexpected format.")

            except Exception as e:
                st.error(f"Error during AI processing: {e}")
                st.exception(e)
