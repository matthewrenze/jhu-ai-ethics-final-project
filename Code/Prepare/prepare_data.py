# Include libraries
import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

# Remove future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Set the file paths
root_folder_path = "C:/Users/Matthew/Dropbox/Personal/School/JHU/Ethics/Project"
input_file_path = root_folder_path + "/Data/Source/compas_synthetic_2023.xlsx"
output_file_path = root_folder_path + "/Data/Prepared/compas_synthetic_2023.csv"
random_ids_file_path = root_folder_path + "/Data/Prepared/random_ids.csv"
shap_mean_values_file_path = root_folder_path + "/Data/Prepared/shap_mean_values.txt"

# Display the initial status
print("Preparing the data...")

# Read the data file
input_data = pd.read_excel(input_file_path)

# Read the random IDs
random_ids = pd.read_csv(random_ids_file_path)

# Get copy the input data
data = input_data.copy()

# Specify the features to use
features = [
    "id",
    "name",
    "decile_score",
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
    "vocational"
]

# Filter the columns
data = data[features]

# Define the features and target variable
target_variable = "decile_score"
X = data.drop(target_variable, axis=1)
y = data[target_variable]

# Exclude id and name from training data
X = X.drop("id", axis=1)
X = X.drop("name", axis=1)

# Encode the categorical variables
X["c_charge_degree"] = X["c_charge_degree"].map({"M": 0, "F": 1})

# Scale all the features from 0 to 1
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
y = scaler.fit_transform(y.values.reshape(-1, 1)).ravel()

# DEBUG: Use all data for training and testing
X_train = X
X_test = X
y_train = y
y_test = y

# Create a random forest regressor
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    random_state=42)

# Train the model
model.fit(X_train, y_train)

# Get the predictions
y_pred = model.predict(X_test)

# Compute the RMSE
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
print(f"RMSE: {rmse:.5f}\n")

# Initialize a TreeExplainer object
explainer = shap.TreeExplainer(model)

# Generate SHAP values for the test set
shap_values = explainer.shap_values(X_test)

# Create a table from the SHAP values
shap_table = pd.DataFrame(shap_values)

# Rename the columns with the suffix "_importance"
shap_table.columns = [
    f"{X_test.columns[i]}_importance"
    for i in range(len(shap_table.columns))]

# Get the absolute SHAP values
shap_values_abs = np.abs(shap_values)

# Create a table from the SHAP values
shap_table_abs = pd.DataFrame(
    data=shap_values_abs,
    columns=X_test.columns)

# Scale all the absolute shap values from 0 to 1 across the columns
# Note: This column-wise scaling; not normal row-wise scaling
scaler = MinMaxScaler(feature_range=(0, 1))
shap_array_scaled = scaler.fit_transform(shap_table_abs.values.T).T

# Convert the scaled values to a table
shap_table_scaled = pd.DataFrame(
    data=shap_array_scaled,
    columns=X_test.columns)

# Rename the columns with the suffix "_importance_scaled"
shap_table_scaled.columns = [
    f"{X_test.columns[i]}_importance_scaled"
    for i in range(len(shap_table_scaled.columns))]

# Round the SHAP values to 3 decimal places
shap_table = shap_table.round(3)
shap_table_scaled = shap_table_scaled.round(3)

# Create importance levels (bins)
def get_level(importance):
    if importance <= 0.33:
        return "low"
    elif importance >= 0.66:
        return "high"
    else:
        return "medium"

# Create importance levels for each feature
importance_levels_table = pd.DataFrame()
for feature in X_test.columns:
    importance_levels_table[feature + "_importance_level"] = shap_table_scaled[feature + "_importance_scaled"].apply(get_level)

# Create risk levels (bins)
def get_level(score):
    if score <= 3:
        return "low"
    elif score >= 6:
        return "high"
    else:
        return "medium"

# Features to level
features_to_level = ["decile_score"] + [X.columns[i] for i in range(len(X.columns))]

# Copy features to level from source data
features_to_level_table = data[features_to_level].copy()

# Encode the categorical variables
features_to_level_table["c_charge_degree"] = features_to_level_table["c_charge_degree"].map({"M": 0, "F": 10})

# Create risk levels for each feature
feature_levels_table = pd.DataFrame()
for feature in features_to_level:
    feature_levels_table[feature + "_level"] = features_to_level_table[feature].apply(get_level)

# Append feature risk levels to the source data
data = pd.concat([
        data.reset_index(drop=True),
        feature_levels_table.reset_index(drop=True)],
    axis=1)

# Concatenate the SHAP values to the source data
data = pd.concat([
        data.reset_index(drop=True),
        shap_table.reset_index(drop=True)],
    axis=1)

# Concatenate the scaled SHAP values to the source data
data = pd.concat([
        data.reset_index(drop=True),
        shap_table_scaled.reset_index(drop=True)],
    axis=1)

# Concatenate the importance levels to the source data
data = pd.concat([
        data.reset_index(drop=True),
        importance_levels_table.reset_index(drop=True)],
    axis=1)

# Get the random IDs
random_ids = random_ids["id"].tolist()

# Filter the data
# NOTE: Need to do this at the end to avoid messing up SHAP values
data = data[data["id"].isin(random_ids)]

# Write the data to a file
data.to_csv(output_file_path, index=False)


### Calculate the mean absolute SHAP values ###

# Calculate the mean absolute SHAP values
shap_values_mean = shap_table_abs.mean(axis=0)

# Scale the values to the interval [0, 1] using min-max scaling
shap_values_mean = MinMaxScaler()\
    .fit_transform(shap_values_mean.values.reshape(-1, 1))\
    .flatten()

# Sort the mean absolute SHAP values
shap_values_mean = pd.Series(
    data=shap_values_mean,
    index=X_test.columns)\
    .sort_values(ascending=False)

# Print the mean absolute SHAP values
for i in range(len(shap_values_mean)):
    print(f"{shap_values_mean.index[i]}: {shap_values_mean[i]:.5f}")
print()

# Write the mean absolute SHAP values to a file
with open(shap_mean_values_file_path, "w") as file:
    for i in range(len(shap_values_mean)):
        file.write(f"{shap_values_mean.index[i]}: {shap_values_mean[i]:.5f}\n")


# Display the final status
print("All data prepared.")