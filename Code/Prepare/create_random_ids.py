# import libraries
import pandas as pd

# Set the file paths
root_folder_path = "C:/Users/Matthew/Dropbox/Personal/School/JHU/Ethics/Project"
compas_file_path = root_folder_path + "/Data/Source/compas_synthetic_2023.xlsx"
random_ids_file_path = root_folder_path + "/Data/Prepared/random_ids.csv"

# Read the data
compas = pd.read_excel(compas_file_path)

# Get the ids
ids = compas["id"]

# Get 100 random ids using a random seed
random_ids = ids.sample(n=100, random_state=42)

# Save the ids to a file
random_ids.to_csv(random_ids_file_path, index=False)
