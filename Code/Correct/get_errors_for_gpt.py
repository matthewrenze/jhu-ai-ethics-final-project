
# NOTE: I will likely need to remove the Results section from this file
# NOTE: Since my rules-based evaluation script does this already

# Import libraries
import os
import pandas as pd

# Set the GPT model
model_name = "gpt-4"
treatment = "corrected-2"

# Set file paths
root_folder_path = "C:/Users/Matthew/Dropbox/Personal/School/JHU/Ethics/Project"
compas_file_path = root_folder_path + "/Data/Prepared/compas_synthetic_2023.csv"
verifications_folder_path = root_folder_path + f"/Data/Verifications/{model_name}-{treatment}"
errors_folder_path = root_folder_path + f"/Data/Errors/{model_name}-{treatment}"

# Specify the features to use
features = [
    "id",
    "name",
    "decile_score",
    "decile_score_level",

    "priors_count",
    "c_charge_degree",
    "juv_fel_count",
    "family_criminality",
    "criminal_attitude",
    "criminal_associates",
    "financial_problems",
    "substance_abuse",
    "noncompliance",
    "social_environment",
    "vocational",

    "priors_count_level",
    "c_charge_degree_level",
    "juv_fel_count_level",
    "family_criminality_level",
    "criminal_attitude_level",
    "criminal_associates_level",
    "financial_problems_level",
    "substance_abuse_level",
    "noncompliance_level",
    "social_environment_level",
    "vocational_level",

    "priors_count_importance_level",
    "c_charge_degree_importance_level",
    "juv_fel_count_importance_level",
    "family_criminality_importance_level",
    "criminal_attitude_importance_level",
    "criminal_associates_importance_level",
    "financial_problems_importance_level",
    "substance_abuse_importance_level",
    "noncompliance_importance_level",
    "social_environment_importance_level",
    "vocational_importance_level",
]


# Display the initial status
print("Evaluating all explanations...")

# Create errors folder if it doesn't exist
if not os.path.exists(errors_folder_path):
    os.makedirs(errors_folder_path)

# Read the data file
data = pd.read_csv(compas_file_path)

# Truncate the name to just first name
# Note: this is because the letters only contain first names
data["name"] = data["name"].str.split(" ").str[0]

# Create a results data frame
results = pd.DataFrame(columns=features)

# Get all file paths
explain_file_paths = os.listdir(verifications_folder_path)

# Verify each explanation file
for explain_file_name in explain_file_paths:

    # Display a status update
    print(f"Evaluating {explain_file_name}...")

    # Get the full file path
    explain_file_path = verifications_folder_path + "/" + explain_file_name

    # Open the input file
    with open(explain_file_path, "r") as explain_file:
        explain_text = explain_file.read()

    # Get the case ID from the file name
    case_id = int(explain_file_name.split("-")[0])

    # Get the case record
    case = data[data["id"] == case_id]

    # Create a function to get the case ID
    def get_feature_value(feature_name, input_text):
        feature_token = f"{feature_name}:"
        start_index = input_text.find(feature_token) + len(feature_token) + 1
        end_index = input_text.find("\n", start_index - 1)
        verify_case_id = input_text[start_index:end_index]
        return verify_case_id

    # Create a function to get the correctness score
    def get_correctness(feature_name, input_text):

        # Get the source and verification values
        source_value = str(case[feature_name].values[0])
        verify_value = get_feature_value(feature_name, input_text)

        # Convert to lower case
        source_value = source_value.lower()
        verify_value = verify_value.lower()

        # Return 0 if the value is empty
        if verify_value == "":
            return 0

        # Return 1 if the values are the same
        return 1 if source_value == verify_value else -1

    # Create a list for errors
    errors = []

    # Get the correctness of all features
    for feature in features:

        # Get the correctness
        correctness = get_correctness(feature, explain_text)

        # Add the correctness to the results
        results.loc[case_id, feature] = correctness

        # If there is an error, then add the error to the list
        if correctness == -1:
            incorrect_value = get_feature_value(feature, explain_text)
            correct_value = str(case[feature].values[0])
            feature_line = f"{feature}: {incorrect_value} -> {correct_value}"
            errors.append(feature_line)

    if len(errors) > 0:
        # Get the errors file name
        errors_file_name = explain_file_name
        errors_file_path = errors_folder_path + "/" + errors_file_name

        # Format the errors file text
        errors_text = "# Errors\n"
        errors_text += "\n".join(errors)

        # Write the errors to a file
        with open(errors_file_path, "w") as errors_file:
            errors_file.write(errors_text)

    # Display a status update
    print(f"Evaluated {explain_file_name}.")

# Display the final status
print("All explanations evaluated.")



