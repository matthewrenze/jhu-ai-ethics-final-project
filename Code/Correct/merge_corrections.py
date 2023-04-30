# Import libraries
import os
import re
import csv
from datetime import datetime
import pandas as pd
import shutil

# Set the model
model_name = "gpt-4"
treatment = "baseline"

# Set file paths
root_folder_path = "C:/Users/Matthew/Dropbox/Personal/School/JHU/Ethics/Project"
explanations_folder_path = root_folder_path + f"/Data/Explanations/{model_name}-{treatment}"
corrections_folder_path = root_folder_path + f"/Data/Corrections/{model_name}-{treatment}"
output_folder_path = root_folder_path + f"/Data/Explanations/{model_name}-corrected"

# Display an initial status
print("Merging all corrections...")

# Copy the explanations folder to the output folder
shutil.copytree(explanations_folder_path, output_folder_path)

# Copy the corrections folder to the output folder (overwrite if exists)
shutil.copytree(corrections_folder_path, output_folder_path, dirs_exist_ok=True)

# Display a final status
print("All corrections merged.")