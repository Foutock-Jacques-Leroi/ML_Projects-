import pandas as pd

def extract_excel_column_to_list(file_path, column_name):
    """
    Reads an Excel file, extracts all text from a specified column,
    and returns it as a list.

    Args:
        file_path (str): The path to the Excel file.
        column_name (str): The name of the column to extract text from.

    Returns:
        list: A list containing all text values from the specified column.
              Returns an empty list if the file or column is not found.
    """
    try:
        # Read the Excel file into a pandas DataFrame
        df = pd.read_excel(file_path)

        # Check if the specified column exists in the DataFrame
        if column_name not in df.columns:
            print(f"Error: Column '{column_name}' not found in the Excel file.")
            return []

        # Extract all values from the specified column
        # Convert all values to string type to ensure we only get text
        column_data = df[column_name].astype(str).tolist()

        return column_data

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

if __name__ == "__main__":
    # --- Configuration ---
    excel_file = "votre_fichier.xlsx"  # Replace with the path to your Excel file
    column_to_extract = "Nom de la colonne"  # Replace with the actual name of your column

    # --- Run the extraction ---
    extracted_texts = extract_excel_column_to_list(excel_file, column_to_extract)

    # --- Display the results ---
    if extracted_texts:
        print(f"Textes extraits de la colonne '{column_to_extract}':")
        for i, text in enumerate(extracted_texts):
            print(f"- Ligne {i+1}: {text}")
    else:
        print("Aucun texte n'a été extrait ou une erreur est survenue.")