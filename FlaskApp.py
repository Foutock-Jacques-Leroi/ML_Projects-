import os
import joblib
import re
from flask import Flask, request
from pydantic import BaseModel





app = Flask(__name__)



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



def Critics_classification():
    input_text = input("Enter your Critics please")
    process_text = preprocess_text(input_text)
    input_vectorized = vector.transform([process_text])
    prediction = model.predict(input_vectorized)
    probas = model.predict_proba(input_vectorized)*100
    pourcentage = probas[0][1]
    if prediction[0] == 1:
        print("The critic is Positive")
        print(f"Pourcentage de resultat Positif : {pourcentage:.2f}%")
    else:
        print("the critic is Negative")
        print(f"Pourcentage de resultat Negatif : {100 - pourcentage:.2f}%")

    return prediction[0]  

@app.route('/', method=["GET"])
def Critics_classification():
    # input_text = input("Enter your Critics please")
    process_text = preprocess_text(input_text)
    input_vectorized = vector.transform([process_text])
    prediction = model.predict(input_vectorized)
    probas = model.predict_proba(input_vectorized)*100
    pourcentage = probas[0][1]
    if prediction[0] == 1:
        print("The critic is Positive")
        print(f"Pourcentage de resultat Positif : {pourcentage:.2f}%")
    else:
        print("the critic is Negative")
        print(f"Pourcentage de resultat Negatif : {100 - pourcentage:.2f}%")

    return prediction[0] 

class Critic_classifier(BaseModel):
    text: str

data = Critic_classifier(**request.json)



if __name__ == "__main__":
    app.run(debug=True)