#import those stuff
import streamlit as st
import pickle
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')

#loading clf n idf
clf = pickle.load(open('clf.pkl','rb'))
tfidf = pickle.load(open('tfidf.pkl','rb'))

# the resume cleaner
def cleanResume(txt):
    # Replace 'RT' or 'cc' with a space
    cleantxt = re.sub(r'\bRT\b|\bcc\b', ' ', txt)
    
    # Remove URLs starting with 'http://'
    cleantxt = re.sub(r'http://\S+', ' ', cleantxt)
    
    # Remove special characters (punctuation and symbols) and '@'
    
    
    # Remove everything from @ onwards (including @)
    cleantxt = re.sub(r'@\S+', ' ', cleantxt)
    
    # Remove hashtags (e.g., #hashtag)
    cleantxt = re.sub(r'#\S+', ' ', cleantxt)
    
    # Remove non-ASCII characters
    cleantxt = re.sub(r'[^\x00-\x7F]+', ' ', cleantxt)
    cleantxt = re.sub(r'\s+',' ',cleantxt)
    # Replace multiple spaces with a single space and strip leading/trailing spaces
    cleantxt = re.sub(r'\s+', ' ', cleantxt).strip()
    cleantxt = re.sub(r'[!"#%&\'()*+,-./:;<=>?@[\]^_`{|}~]+', ' ', cleantxt)
    return cleantxt

#the codes for each title
category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }
#the web app

def main():

    st.title("Resume classifing App")
    uploaded_file =st.file_uploader('Upload Resume',type=['txt','pdf'])
    #check if file is uploaded or not
    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')
        clean_resume = cleanResume(resume_text)
        clean_resume=tfidf.transform([clean_resume])
        prediction_id = clf.predict(clean_resume)[0]
        category = category_mapping.get(prediction_id,"Not known")
        st.write("You are predicted as :",category)


if __name__ == "__main__":
    main()