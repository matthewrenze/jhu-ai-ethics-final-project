# Import libraries
import os
import re
import pandas as pd
from fuzzywuzzy import process

# Set the GPT model
model_name = "gpt-4"
treatment = "corrected"

# Set the file paths
root_folder_path = "C:/Users/Matthew/Dropbox/Personal/School/JHU/Ethics/Project"
explanation_folder_path = root_folder_path + f"/Data/Explanations/{model_name}-{treatment}"
compas_file_path = root_folder_path + "/Data/Prepared/compas_synthetic_2023.csv"
results_file_path = root_folder_path + f"/Data/Results/{model_name}-{treatment}.csv"

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

feature_description_map = {
    "priors_count": "number of prior offenses",
    "c_charge_degree": "charge degree",
    "juv_fel_count": "number of juvenile felonies",
    "family_criminality": "family criminality",
    "criminal_attitude": "criminal attitude",
    "criminal_associates": "criminal associates",
    "financial_problems": "financial problems",
    "substance_abuse": "substance abuse",
    "noncompliance": "noncompliance",
    "social_environment": "social environment risk",
    "vocational": "vocational risk factors"}

# Create a function to verify the name
def verify_name(expected, line):
    actual = line.split(",")[0]
    return 1 if actual == expected else -1

# Create a function to verify the case ID
def verify_case_id(expected, line):
    actual = int(re.search(r"#(\d+)", line).group(1))
    return 1 if actual == expected else -1

# Create a function to verify the decile score
def verify_decile_score(expected, line):
    actual = int(re.search(r"COMPAS risk score of (\d+)", line).group(1))
    return 1 if actual == expected else -1

# Create a function to verify the decile score level
def verify_decile_score_level(expected, line):
    actual = re.search(r"scored as a (\w+) risk", line).group(1)
    return 1 if actual == expected else -1

# Create a function to fuzzy-match a feature description
def fuzzy_match_feature(feature_description):
    feature_descriptions = feature_description_map.values()
    matches = process.extractOne(feature_description, feature_descriptions)
    best_match = matches[0]
    best_feature = [k for k, v in feature_description_map.items() if v == best_match][0]
    return best_feature

def get_features(line_part):
    reg_ex = r"(high|medium|low) risk score for (.+?) \((\d+)\)"
    match = re.search(reg_ex, line_part)
    has_match = match is not None
    level = match.group(1) if match else None
    feature_description = match.group(2) if match else None
    feature = fuzzy_match_feature(feature_description) if match else None
    score = int(match.group(3)) if match else None
    end_index = match.end() if match else None

    return has_match, feature, score, level, end_index

# Create a function to get low-importance features
def get_low_importance_features(line_part):
    reg_ex = r"(?:including |, and? |, )(.+?) \((\d+)\)"
    match = re.search(reg_ex, line_part)
    has_match = match is not None
    feature_description = match.group(1) if match else None
    feature = fuzzy_match_feature(feature_description) if match else None
    score = int(match.group(2)) if match else None
    end_index = match.end() if match else None

    return has_match, feature, score, end_index

def verify_value(expected, actual):
    return 1 if actual == expected else -1

# Display the initial status
print("Evaluating all explanations...")

# Get all files in the source folder
explanation_file_names = os.listdir(explanation_folder_path)

# Read the compas data
compas_data = pd.read_csv(compas_file_path)

# Create a results data frame
results = pd.DataFrame(columns=features)

# Verify each explanation file
for explain_file_name in explanation_file_names:

    # Display a status update
    print(f"Evaluating {explain_file_name}...")

    # Get the case ID from the first part of the file name
    case_id = int(explain_file_name.split("-")[0])

    # Get the case from the compas data
    case = compas_data.loc[compas_data["id"] == case_id]

    # Get the path to the explanation file
    explain_file_path = explanation_folder_path + "/" + explain_file_name

    # Read the letter of explanation file
    with open(explain_file_path, "r") as explanation_file:
        explanation_lines = explanation_file.readlines()

    # Set all features to 0
    for feature in features:
        results.loc[case_id, feature] = 0

    for line in explanation_lines:

        # If it's the greeting line
        if explanation_lines.index(line) == 0:
            full_name = case["name"].values[0]
            first_name = full_name.split(" ")[0]
            first_name = first_name.title()
            correctness = verify_name(first_name, line)
            results.loc[case_id, "name"] = correctness

        # If it's the case info line
        elif line.startswith("Regarding your case"):

            # Verify the case ID
            case_id_correctness = verify_case_id(case_id, line)
            results.loc[case_id, "id"] = case_id_correctness

            # Verify the decile score
            decile_score = case["decile_score"].values[0]
            decile_score_correctness = verify_decile_score(decile_score, line)
            results.loc[case_id, "decile_score"] = decile_score_correctness

            # Verify the decile score level
            decile_score_level = case["decile_score_level"].values[0]
            decile_score_level_correctness = verify_decile_score_level(decile_score_level, line)
            results.loc[case_id, "decile_score_level"] = decile_score_level_correctness

        # If it's the high- or medium-importance line
        elif line.startswith("You received this score") or line.startswith("You also had a"):
            line_part = line
            has_match = True
            end_index = 0
            while has_match:
                actual_feature_details = get_features(line_part)
                has_match, actual_feature, actual_score, actual_level, end_index = actual_feature_details
                if has_match:
                    expected_score = case[actual_feature].values[0]
                    score_correctness = verify_value(expected_score, actual_score)
                    results.loc[case_id, actual_feature] = score_correctness

                    level_name = f"{actual_feature}_level"
                    expected_level = case[level_name].values[0]
                    level_correctness = verify_value(expected_level, actual_level)
                    results.loc[case_id, level_name] = level_correctness

                    importance_name = f"{actual_feature}_importance_level"
                    actual_importance = "high" \
                        if line.startswith("You received this score") \
                        else "medium"
                    expected_importance = case[importance_name].values[0]
                    importance_correctness = verify_value(expected_importance, actual_importance)
                    results.loc[case_id, importance_name] = importance_correctness

                    line_part = line_part[end_index:]

        # If it's the low-importance line
        elif line.startswith("The remaining risk factors"):
            line_part = line
            has_match = True
            end_index = 0
            while has_match:
                actual_feature_details = get_low_importance_features(line_part)
                has_match, actual_feature, actual_score, end_index = actual_feature_details
                if has_match:
                    expected_score = case[actual_feature].values[0]
                    score_correctness = verify_value(expected_score, actual_score)
                    results.loc[case_id, actual_feature] = score_correctness

                    importance_name = f"{actual_feature}_importance_level"
                    actual_importance = "low"
                    expected_importance = case[importance_name].values[0]
                    importance_correctness = verify_value(expected_importance, actual_importance)
                    results.loc[
                        case_id, importance_name] = importance_correctness

                    line_part = line_part[end_index:]

    # Display a status update
    print(f"Evaluation complete.")

# Convert the index to a column called "case_id"
results.reset_index(inplace=True)
results.rename(columns={"index": "case_id"}, inplace=True)

# Save the results
results.to_csv(results_file_path, index=False)

# Display the final status
print("All explanations verified.")