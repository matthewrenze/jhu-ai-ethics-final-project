# Import libraries
import pandas as pd
import shutil

# Set the file paths
root_folder_path = "C:/Users/Matthew/Dropbox/Personal/School/JHU/Ethics/Project"
compas_file_path = root_folder_path + "/Data/Prepared/compas_synthetic_2023.csv"
random_ids_file_path = root_folder_path + "/Data/Prepared/random_ids.csv"
cases_folder_path = root_folder_path + "/Data/Cases"
example_cases_folder_path = root_folder_path + "/Data/Examples/Cases"

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

# Display initial status
print("Preparing all cases...")

# Read the data file
data = pd.read_csv(compas_file_path)

# Read the random ids file
random_ids = pd.read_csv(random_ids_file_path)

# For each case
for case_id in random_ids["id"].values:

    # Display a status update
    print(f"Preparing case {case_id}...")

    # Get the case record
    case = data[data["id"] == case_id]

    # Get the defendant's  name
    defendant_name = case["name"].values[0]

    # Add the case info to the case record
    case_record = "# Case Info\n"
    for feature in case_info:
        feature_value = case[feature].values[0]
        case_record += f"{feature}: {feature_value}\n"
    case_record += "\n"

    # Add the features to the case record
    case_record += "# Features\n"
    for feature in features:
        feature_value = case[feature].values[0]
        case_record += f"{feature}: {feature_value}\n"
    case_record += "\n"

    # Add the feature levels to the case record
    case_record += "# Feature Levels\n"
    for feature in feature_levels:
        feature_value = case[feature].values[0]
        case_record += f"{feature}: {feature_value}\n"
    case_record += "\n"

    # Create a new dataframe for the feature importance
    importance_scaled_table = pd.DataFrame(columns=["feature", "value"])

    # Add the feature importance to the dataframe
    for feature in feature_importance_scaled:
        feature_value = case[feature].values[0]
        importance_scaled_table = importance_scaled_table._append(
            {"feature": feature, "value": feature_value},
            ignore_index=True)

    # Sort the scaled feature importance by value
    importance_scaled_table = importance_scaled_table\
        .sort_values(by="value", ascending=False)

    # Create a new dataframe for the feature importance levels
    importance_levels_table = pd.DataFrame(columns=["feature", "value"])

    # Add the importance levels to the dataframe
    for feature in feature_importance_levels:
        feature_value = case[feature].values[0]
        importance_levels_table = importance_levels_table._append(
            {"feature": feature, "value": feature_value},
            ignore_index=True)

    # Sort the importance levels using the index of the scaled importance values
    importance_levels_table = importance_levels_table\
        .reindex(importance_scaled_table.index)

    # Add the importance levels to the case record
    case_record += "# Feature Importance Levels\n"
    for index, row in importance_levels_table.iterrows():
        feature = row["feature"]
        feature_value = row["value"]
        case_record += f"{feature}: {feature_value}\n"
    case_record += "\n"

    # Create the case record file path
    defendant_name = defendant_name.replace(" ", "-")
    case_file_name = f"{case_id}-{defendant_name}.txt"
    case_file_path = cases_folder_path + "/" + case_file_name

    # Write the case record to a file
    with open(case_file_path, "w") as case_file:
        case_file.write(case_record)

    # Display a status update
    print(f"Case {case_id} prepared.")

# Display final status
print("All cases prepared.")

# Display initial status
print("Copying example cases...")
