import streamlit as st
import os
import joblib
import re
import pandas as pd

from streamlit_option_menu import option_menu





st.set_page_config(
    layout="wide",
    page_title="Critics App",
    page_icon="🎬",
    initial_sidebar_state="auto"
)
# st.sidebar.success("hello")
selected = option_menu(
    menu_title=None,
    options=["Home Page", "Bulk Analysis"],
    default_index=0,
orientation="horizontal"
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


def preprocess_text(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split() # This creates a list of words
    review = [word for word in review if word not in all_stopwords]
    review = ' '.join(review) # THIS IS THE CRUCIAL STEP THAT MAKES IT A STRING AGAIN
    return review

# pourcentage=0
# prediction=0


    # input_text = input("Enter your Critics please")


# Critics_classification()

if selected == "Home Page":

    st.title("Movies and Series Critics sentiments prediction")
    c1, c2, c3 = st.columns(3)

    with c2:
        st.info("**If you are neutral your critics is of no need !**")

    st.subheader("Critics about any movie ")

    text_from_user = st.text_area("**Enter Your Point Of View**", height=200)


    # process_text = preprocess_text(text_from_user)
    # input_vectorized = vector.transform([process_text])
    # prediction = model.predict(input_vectorized)

    # probas = model.predict_proba(input_vectorized)*100
    # pourcentage = probas[0][1]

    # process_text = preprocess_text(text_from_user)
    # input_vectorized = vector.transform([process_text])
    # prediction = model.predict(input_vectorized)
    # probas = model.predict_proba(input_vectorized)*100
    # pourcentage = probas[0][1]

    # if prediction[0]:
    #     print(f"The critic is Positive a {pourcentage:.2f}%")
    #     # print(f"Pourcentage de resultat Positif : {pourcentage:.2f}%")
    # else:
    #     print(f"the critic is Negative a {100 - pourcentage:.2f}%")
    #     # print(f"Pourcentage de resultat Negatif : {pourcentage:.2f}%")



    button = st.button("**Critics Analysis Result**", key="critics")
    # button = st.button("Critics Analysis Result", key="hello")

    # col1, col2 = st.columns(2)
    cc1,cc2,cc3 = st.columns(3)

    with cc2:
        if button:
            process_text = preprocess_text(text_from_user)
            input_vectorized = vector.transform([process_text])
            prediction = model.predict(input_vectorized)
            print(prediction)
            probas = model.predict_proba(input_vectorized)*100
            pourcentage = probas[0][1]

            if prediction[0] == 1:
                
                st.metric(label="Pourcentages Precision of the Model :", value=f"{pourcentage:.2f}%")
                st.success("You Get A Positive Critics Congratulation! 🎉✨")
                
            else:   
                st.metric(label="Pourcentages Precision of the Model :", value=f"{100 - pourcentage:.2f}%")
                st.error("You Get A Negative Critics... 😣😑")

if selected == "Bulk Analysis":
    
    st.title(" Dataset Sentiment Classifier.")


    col_name = st.text_input("**Enter the target Column** ")

    file = st.file_uploader("Enter your file")
    try:
        if file:
            
            df = pd.read_csv(file)
            if not type(df):
                
                st.header("comm")
            # col_name = input("enter the target column:  ")

            # print(df["Text"].head())
            # 
            else:

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

                for i in df[col_name]:
                    array_str.append(i)

                progress_bar.progress(30)
                

                def preprocess_text(text):
                    review = re.sub('[^a-zA-Z]', ' ', text)
                    review = review.lower()
                    review = review.split() # This creates a list of words
                    review = [word for word in review if word not in all_stopwords]
                    review = ' '.join(review) # THIS IS THE CRUCIAL STEP THAT MAKES IT A STRING AGAIN
                    return review


                progress_bar.progress(50)

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

                progress_bar.progress(80)
                real_p = len(positif)
                real_n = len(negatif)
                total = real_p + real_n

                progress_bar.progress(100)



                if progress_bar.progress(100):
                    result = st.success(" Processing complete !")
                    
                

                    st.bar_chart([len(negatif), len(positif)], y_label="Critics Average size", x_label="Final Results")

                    st.divider()

                    s1, s2, s3, s4, s5 = st.columns(5)

                    with s3:
                        st.header("Critics Results")

                    q1, q2, q3, q4, q5, q6 = st.columns(6)
                    st.dataframe(df.head())

                    with q2:
                        st.metric("**Positif Critics**", real_p, total)
                    with q5:
                        st.metric("**Negatif Critics**", real_n, total)

                st.divider()

    except KeyError:
        st.header("Column don't exist.  **Retry**")
        st.error("No Match found !")