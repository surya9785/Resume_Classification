import streamlit as st
import pickle
import re
import nltk
from docx import Document  # Import for handling .docx files

nltk.download('punkt')
nltk.download('stopwords')

# Loading models
clf = pickle.load(open('clf.pkl', 'rb'))
count = pickle.load(open('count.pkl', 'rb'))


def clean_resume(resume_text):
    """
    Cleans the resume text by removing special characters, URLs, and other noise.
    """
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', ' ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text


def clean_docx_resume(docx_file):
    """
    Extracts text from a .docx file and cleans it using the clean_resume function.
    """
    doc = Document(docx_file)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    resume_text = '\n'.join(full_text)
    return clean_resume(resume_text)


def main():
    st.title("Resume Screening App")
    uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'pdf', 'docx'])
    try:
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.docx'):
                    resume_text = clean_docx_resume(uploaded_file)
                else:
                    resume_bytes = uploaded_file.read()
                    resume_text = resume_bytes.decode('utf-8')
                    # Attempt 'latin-1' decoding if UTF-8 fails (as in previous code)
            except UnicodeDecodeError:
                resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = clean_resume(resume_text)
        input_features = count.transform([cleaned_resume])
        prediction_id = clf.predict(input_features)[0]
        #st.write(prediction_id)

            # Map category ID to category name
        category_map = {
                0: "Peoplesoft resumes",
                1: "React Developer",
                2: "SQL developer",
                3: "Workday"
            }

        category_name = category_map.get(prediction_id, "Unknown")
        st.write("Predicted Category:", category_name) 
    except Exception as e:
        print("Please Upload The File")
        #st.error("Error processing resume. Please try again.")


# Python main
if __name__ == "__main__":
    main()
