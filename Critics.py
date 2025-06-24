import streamlit as st
import os
import joblib
import re



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

st.title("Movies and Series Critics sentiments prediction")
c1, c2, c3 = st.columns(3)

with c2:
    st.info("**If you are neutral your critics is of no need !**")

st.subheader("Critics about any movie ")

text_from_user = st.text_area("**Enter Your Point Of View**", height=200)


process_text = preprocess_text(text_from_user)
input_vectorized = vector.transform([process_text])
prediction = model.predict(input_vectorized)

probas = model.predict_proba(input_vectorized)*100
pourcentage = probas[0][1]

if prediction[0] == 1:
    print(f"The critic is Positive a {pourcentage:.2f}%")
    # print(f"Pourcentage de resultat Positif : {pourcentage:.2f}%")
else:
    print(f"the critic is Negative a {100 - pourcentage:.2f}%")
    # print(f"Pourcentage de resultat Negatif : {pourcentage:.2f}%")



button = st.button("**Critics Analysis Result**", key="critics")
# button = st.button("Critics Analysis Result", key="hello")

# col1, col2 = st.columns(2)
cc1,cc2,cc3 = st.columns(3)

with cc2:
    if button:

        if prediction[0] == 1:
            
            st.metric(label="Pourcentages Precision of the Model :", value=f"{pourcentage:.2f}%")
            st.success("You Get A Positive Critics Congratulation! 🎉✨")
            
        else:
            
            
            st.metric(label="Pourcentages Precision of the Model :", value=f"{100 - pourcentage:.2f}%")
            st.error("You Get A Negative Critics... 😣😑")

