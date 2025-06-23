# import streamlit as st
import joblib
import os # Good practice for path handling





# Define the path to your .joblib file
# Let's assume your model is named 'my_model.joblib' and is in the 'model' directory
model_directory = "C:/Users/DELL/Desktop/Memoire Licence/Projet_PPPE/Projet/model"
model_filename = 'Logistic_Regression_model_updated.joblib' # Or 'classifier.joblib', 'tfidf_vectorizer.joblib' etc.
model_path = os.path.join(model_directory, model_filename)

# Check if the file exists before attempting to load it (good practice)
if not os.path.exists(model_path):
    print(f"Error: The model file '{model_path}' was not found.")
else:
    try:
        # Load the model
        loaded_object = joblib.load(model_path)

        print(f"Object '{model_filename}' loaded successfully!")
        print(f"Type of loaded object: {type(loaded_object)}")

        # Now you can use your loaded_object
        # If it's a scikit-learn model:
        # predictions = loaded_object.predict(some_new_data)

        # If it's a vectorizer:
        # transformed_text = loaded_object.transform(["some text"]).toarray()
        # print(transformed_text[0])

    except Exception as e:
        print(f"An error occurred while loading the object: {e}")
        print("Possible reasons: The file might be corrupted, incomplete, or saved with an incompatible joblib/Python version.")



# Define the path to your .joblib file
# Let's assume your model is named 'my_model.joblib' and is in the 'model' directory
model_directory = "C:/Users/DELL/Desktop/Memoire Licence/Projet_PPPE/Projet/model"
model_filename = 'Vectorizer_model_updated.joblib' # Or 'classifier.joblib', 'tfidf_vectorizer.joblib' etc.
model_path = os.path.join(model_directory, model_filename)

# Check if the file exists before attempting to load it (good practice)
if not os.path.exists(model_path):
    print(f"Error: The model file '{model_path}' was not found.")
else:
    try:
        # Load the model
        loaded_object = joblib.load(model_path)

        print(f"Object '{model_filename}' loaded successfully!")
        print(f"Type of loaded object: {type(loaded_object)}")

        # Now you can use your loaded_object
        # If it's a scikit-learn model:
        # predictions = loaded_object.predict(some_new_data)

        # If it's a vectorizer:
        # transformed_text = loaded_object.transform(["some text"]).toarray()
        # print(transformed_text[0])

    except Exception as e:
        print(f"An error occurred while loading the object: {e}")
        print("Possible reasons: The file might be corrupted, incomplete, or saved with an incompatible joblib/Python version.")





# load_model = pickle.load("Logistic_Regression_model.pkl")
# vectorizer = pickle.load("Vectorizer_text.pkl")


# def Critics_classification():
#     input_text = [input("Enter your Critics please")]
#     input_vectorized = vectorizer.transform([input_text])
#     prediction = load_model.predict(input_vectorized)
#     return prediction[0]  


# Critics_classification()