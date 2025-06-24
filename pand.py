import streamlit as st
import pandas as pd
import re
import os
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Bar Chart Visualizer (Matplotlib & Seaborn)",
    layout="centered",
    initial_sidebar_state="expanded"
)

model = os.path.join("C:/Users/DELL/Desktop/Memoire Licence/Projet_PPPE/Projet/model/Logistic_Regression_model_updated.joblib")
# print(model)
vectorizer = os.path.join("C:/Users/DELL/Desktop/Memoire Licence/Projet_PPPE/Projet/model/Vectorizer_model_updated.joblib")
# print(vectorizer)

model = joblib.load(model)
# print(f"Type of loaded object: {type(model)}")
# print(model)

vector = joblib.load(vectorizer)
# print(f"Type of loaded object: {type(vector)}")




st.title("testing import ")

file = st.file_uploader("Enter your file")

if file:
    df = pd.read_csv(file)
# col_name = input("enter the target column:  ")

# print(df["Text"].head())
# 
st.dataframe(df.head(3))

st.header("**PROCESSING ANALYSIS ...**")

progress_bar = st.progress(0)




all_stopwords = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
])

progress_bar.progress(10)

array_str = []

for i in df["Text"]:
    array_str.append(i)

progress_bar.progress(20)
print(array_str)

def preprocess_text(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split() # This creates a list of words
    review = [word for word in review if word not in all_stopwords]
    review = ' '.join(review) # THIS IS THE CRUCIAL STEP THAT MAKES IT A STRING AGAIN
    return review


progress_bar.progress(40)

# p=0
# n=0
positif = []
negatif = []

for i in array_str:
    process_text = preprocess_text(i)
    input_vectorized = vector.transform([process_text])
    prediction = model.predict(input_vectorized)
    probas = model.predict_proba(input_vectorized)*100
    pourcentage = probas[0][1]
    if prediction[0] == 1:
        positif.append("Positive")
    else:
        negatif.append("Negative")

progress_bar.progress(60)
real_p = len(positif)
real_n = len(negatif)
total = real_p + real_n


print(positif)
print(negatif)

progress_bar.progress(100)



if progress_bar.progress(100):
    result = st.success(" Processing complete !")
    
   

    st.bar_chart([len(negatif), len(positif)], y_label="Critics Average size", x_label="Final Results")

    st.divider()

    s1, s2, s3 = st.columns(3)

    with s2:
        st.header("Critics Results")

    q1, q2, q3, q4, q5 = st.columns(5)

    with q2:
        st.metric("**Positif Critics**", real_p, total)
    with q4:
        st.metric("**Negatif Critics**", real_n, total)

    st.divider()   