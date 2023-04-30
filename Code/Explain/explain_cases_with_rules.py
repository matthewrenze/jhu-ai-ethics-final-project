# Import libraries
import os
import csv
from datetime import datetime
import pandas as pd

# Set the model
model_name = "rules"
treatment = "baseline"
task = "explain"

# Set file paths
root_folder_path = "C:/Users/Matthew/Dropbox/Personal/School/JHU/Ethics/Project"
template_file_path = root_folder_path + "/Data/Templates/explain-cases-with-rules-template.txt"
compas_file_path = root_folder_path + "/Data/Prepared/compas_synthetic_2023.csv"
explanations_folder_path = root_folder_path + f"/Data/Explanations/{model_name}-{treatment}"
example_cases_folder_path = root_folder_path + "/Data/Examples/Cases"
log_file_name = f"{model_name}-{treatment}-{task}.csv"
log_folder_path = root_folder_path + f"/Data/Logs"
log_file_path = log_folder_path + "/" + log_file_name

# Specify the features to use
case_info = [
    "id",
    "name",
    "decile_score",
    "decile_score_level"]

features = [
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
    "vocational"]

feature_levels = [
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
    "vocational_level"]

feature_importance_scaled = [
    "priors_count_importance_scaled",
    "c_charge_degree_importance_scaled",
    "juv_fel_count_importance_scaled",
    "family_criminality_importance_scaled",
    "criminal_attitude_importance_scaled",
    "criminal_associates_importance_scaled",
    "financial_problems_importance_scaled",
    "substance_abuse_importance_scaled",
    "noncompliance_importance_scaled",
    "social_environment_importance_scaled",
    "vocational_importance_scaled"]

feature_importance_levels = [
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
    "vocational_importance_level"]

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

written_numbers_map = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten"}

# Create a function to get the written number of an integer
def get_written_number(number):
    if number > 10:
        return str(number)
    return written_numbers_map[str(number)]

# Create a function to create the feature phrases
def create_feature_phrases(features_sorted, importance_level, phrase_template):
    feature_phrases = []
    for feature_name in features_sorted:
        feature_level_name = f"{feature_name}_level"
        feature_importance_level_name = f"{feature_name}_importance_level"
        if case[feature_importance_level_name] == importance_level:
            feature_level = case[feature_level_name]
            feature_description = feature_description_map[feature_name]
            feature_score = case[feature_name]
            phrase = phrase_template.format(
                level=feature_level,
                description=feature_description,
                score=feature_score)
            feature_phrases.append(phrase)
    return feature_phrases

# Create a function to create a text string of conjoined feature phrases
def create_phrases_text(feature_phrases, feature_importance_level):
    if len(feature_phrases) == 0:
        return f"no {feature_importance_level} importance risk factors"
    elif len(feature_phrases) == 1:
        return feature_phrases[0]
    elif len(feature_phrases) == 2:
        return f"{feature_phrases[0]} and {feature_phrases[1]}"
    else:
        return ", ".join(feature_phrases[:-1]) + f", and {feature_phrases[-1]}"

# Create a function to get the high importance follow-up sentence
def get_high_followup_sentence(phrases, decile_score_level):
    if len(phrases) == 0:
        return ""
    elif len(phrases) == 1:
        return f"This was the most important factor in receiving a {decile_score_level} overall risk score."
    else:
        number = written_numbers_map[len(phrases)]
        return f"These were the {number} most important factors in receiving a {decile_score_level} overall risk score."

def get_medium_followup_sentence(phrases):
    if len(phrases) == 0:
        return ""
    elif len(phrases) == 1:
        return f"However, this factor only contributed medium importance to the COMPAS algorithm's scoring of your overall risk."
    else:
        return f"However, these factors only contributed medium importance to the COMPAS algorithm's scoring of your overall risk."


# Display the initial status
print("Explaining all cases with rules...")

# Create output directory if it doesn't exist
if not os.path.exists(explanations_folder_path):
    os.makedirs(explanations_folder_path)

# Create the log folder if it does not exist
if not os.path.exists(os.path.dirname(log_file_path)):
    os.makedirs(os.path.dirname(log_file_path))

# Create a CSV header for the log file
log_header = [
    "Date-Time",
    "Model",
    "Treatment",
    "Task",
    "Prompt Tokens",
    "Completion Tokens",
    "Total Tokens",
    "Start Date-Time",
    "End Date-Time",
    "Duration (s)",
    "Prompt",
    "Response",
    "Error Message"]

# Write the header to the log file
with open(log_file_path, "a", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(log_header)

# Open the data file
data = pd.read_csv(compas_file_path)

# Read the template file
with open(template_file_path, "r") as template_file:
    template = template_file.read()

# For each case in the data
for index, case in data.iterrows():

    # Get the case id
    case_id = case["id"]

    # Display a status update
    print(f"Explaining case {case_id} with rules...")

    # Get the start time
    start_time = datetime.now()

    # Get the case info
    defendant_name = case["name"]
    first_name = defendant_name.split(" ")[0].title()
    decile_score = case["decile_score"]
    decile_score_level = case["decile_score_level"]

    # Sort the features by their importance
    features_sorted = sorted(
        features,
        key=lambda x: case[f"{x}_importance_scaled"],
        reverse=True)

    # Create the feature phrase templates
    high_phrase_template = "a {level} risk score for {description} ({score})"
    medium_phrase_template = "a {level} risk score for {description} ({score})"
    low_phrase_template = "{description} ({score})"

    # Create the feature phrases
    high_phrases = create_feature_phrases(features_sorted, "high", high_phrase_template)
    medium_phrases = create_feature_phrases(features_sorted, "medium", medium_phrase_template)
    low_phrases = create_feature_phrases(features_sorted, "low", low_phrase_template)

    # Only take the top 3 low phrases
    low_phrases = low_phrases[:3]

    # Create the feature phrases text
    high_features = create_phrases_text(high_phrases, "high")
    medium_features = create_phrases_text(medium_phrases, "medium")
    low_features = create_phrases_text(low_phrases, "low")

    # Create follow-up sentences
    high_followup_sentence = get_high_followup_sentence(high_phrases, decile_score_level)
    medium_followup_sentence = get_medium_followup_sentence(medium_phrases)

    # Create the replacements for the template placeholders
    replacements = {}
    replacements["case_id"] = case_id
    replacements["first_name"] = first_name
    replacements["decile_score"] = decile_score
    replacements["decile_score_level"] = decile_score_level
    replacements["high_features"] = high_features
    replacements["high_followup_sentence"] = high_followup_sentence
    replacements["medium_features"] = medium_features
    replacements["medium_followup_sentence"] = medium_followup_sentence
    replacements["low_features"] = low_features

    # Replace the placeholders in the template
    explanation_text = template.format(**replacements)

    # Create the explanation file path
    defendant_name = defendant_name.replace(" ", "-")
    explanation_file_name = f"{case_id}-{defendant_name}.txt"
    explanation_file_path = explanations_folder_path + "/" + explanation_file_name

    # Write the explanation to a text file
    with open(explanation_file_path, "w") as f:
        f.write(explanation_text)

    # Get the end time
    end_time = datetime.now()

    # Get the duration
    duration = end_time - start_time

    # Create the log row
    log_row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        model_name,
        treatment,
        task,
        "0",
        "0",
        "0",
        start_time.strftime("%Y-%m-%d %H:%M:%S"),
        end_time.strftime("%Y-%m-%d %H:%M:%S"),
        duration.total_seconds(),
        "",
        "",
        ""]

    # Write the log row to the log file
    with open(log_file_path, "a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(log_row)

    # Display a status update
    print(f"Case {case_id} explained.")

# Display the final status
print("All cases explained.")