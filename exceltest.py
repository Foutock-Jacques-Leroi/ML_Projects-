import streamlit as st


import pandas as pd
import tempfile
import os




st.title("File Import ")
st.header("upload your file :")
file = st.file_uploader("enter you file",  label_visibility="visible", width="stretch")

original_filename = file.name
st.info(f"Original filename: {original_filename}")
        # 2. Create a temporary directory and save the file
with tempfile.TemporaryDirectory() as temp_dir:
    # Construct the full temporary path
    temp_file_path = os.path.join(temp_dir, original_filename)
    print(temp_file_path)

    # Write the uploaded file's content to the temporary path
    with open(temp_file_path, "wb") as f:
        f.write(file.getvalue())
        # print(file.getvalue())
print(file)
column_to_extract = st.text_input(
    "Enter the exact name of the column to extract:",
    "YourColumnName",
    help="This is case-sensitive. E.g., 'Product_ID' or 'Description'."
)
with st.spinner(f"Extracting text from temporary file: {temp_file_path}"):
    # Read the CSV using the temporary path
    df_from_path = pd.read_csv(temp_file_path)

    if column_to_extract not in df_from_path.columns:
        st.error(f"Error: Column '{column_to_extract}' not found in the CSV file at `{temp_file_path}`.")
    else:
        extracted_texts = df_from_path[column_to_extract].astype(str).tolist()

        if extracted_texts:
            st.success(f"Successfully extracted {len(extracted_texts)} text entries from column '{column_to_extract}'.")
            st.subheader("Extracted Texts:")
            st.dataframe(pd.DataFrame(extracted_texts, columns=[column_to_extract]))
        else:
            st.warning("No texts were extracted or column was empty.")
