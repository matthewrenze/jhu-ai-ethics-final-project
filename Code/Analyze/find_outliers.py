# Import libraries
import pandas as pd

# Set the file paths
root_folder_path = "C:/Users/Matthew/Dropbox/Personal/School/JHU/Ethics/Project"
compas_file_path = root_folder_path + "/Data/Prepared/compas_synthetic_2023_full.csv"

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

# Read the compas data
data = pd.read_csv(compas_file_path)

# Create a function to find a case with no high-importance features
def find_cases_with(importance_level, feature_count):
    results = []
    for index, row in data.iterrows():
        current_feature_count = 0
        for feature_importance_level in feature_importance_levels:
            if row[feature_importance_level] == importance_level:
                current_feature_count += 1
        if current_feature_count == feature_count:
            case_id = row["id"]
            name = row["name"].replace(" ", "-")
            result = f"{case_id}-{name}.txt"
            results.append(result)
    return results


# Create a function to print the first result
def print_one(importance_level, feature_count):
    case = f"{importance_level}={feature_count}"
    results = find_cases_with(importance_level, feature_count)
    if results is None or len(results) == 0:
        print(f"{case}: None")
    else:
        print(f"{case}: {results[0]}")

# Print a single outlier for each case
print_one("high", 0)
print_one("high", 1)
print_one("high", 2)
print_one("high", 3)
print_one("high", 4)
print_one("high", 5)
print()

print_one("medium", 0)
print_one("medium", 1)
print_one("medium", 2)
print_one("medium", 3)
print_one("medium", 4)
print_one("medium", 5)
print()

print_one("low", 0)
print_one("low", 1)
print_one("low", 2)
print_one("low", 3)
print_one("low", 4)
print_one("low", 5)
print()

# Find a case with a decile score of 10
score_10 = data[data["decile_score"] == 10]
print(f"Score 10: {score_10['id'].values[0]}-{score_10['name'].values[0].replace(' ', '-')}.txt")


