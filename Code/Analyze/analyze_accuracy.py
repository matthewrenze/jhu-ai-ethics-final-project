# Import libraries
import os
import pandas as pd
import matplotlib.pyplot as plt

# Set the file paths
root_folder_path = "C:/Users/Matthew/Dropbox/Personal/School/JHU/Ethics/Project"
results_folder_path = root_folder_path + "/Data/Results"
analysis_folder_path = root_folder_path + "/Data/Analysis"

# Create an empty data frame
data = pd.DataFrame()

# Get the results files
results_file_names = []
for results_file_name in os.listdir(results_folder_path):
    if results_file_name.endswith(".csv"):
        results_file_names.append(results_file_name)

# Loop through the results files
for results_file_name in results_file_names:

    # Get the file path
    results_file_path = results_folder_path + "/" + results_file_name

    # Read the data
    data_to_append = pd.read_csv(results_file_path)

    # Get the model and treatment
    model_name_parts = results_file_name.split("-")[0:-1]
    model_name = "-".join(model_name_parts)
    treatment = results_file_name.split("-")[-1].split(".")[0]

    # Insert the model and treatment
    data_to_append.insert(0, "Model", model_name)
    data_to_append.insert(1, "Treatment", treatment)

    # Append the data
    data = data._append(data_to_append, ignore_index=True)

# Copy the data
data_to_summarize = data.copy()

# Remove the case_id column
data_to_summarize = data_to_summarize.drop(columns=['case_id'])

# Rename "id" column
data_to_summarize = data_to_summarize.rename(columns={"id": "case_id"})

# Group the data by 'Model' and 'Treatment'
grouped_data = data_to_summarize.groupby(['Model', 'Treatment'])

# Define unique_values
unique_values = [-1, 0, 1]

# Create an empty list to store the aggregated data
summary_data = []

# Loop through each group and count correctness values
for (model, treatment), group in grouped_data:
    for value in unique_values:
        value_counts = (group == value).sum()
        row = {'Value': value, 'Model': model, 'Treatment': treatment, **value_counts.to_dict()}
        row["Model"] = model
        row["Treatment"] = treatment
        summary_data.append(row)

# Create a summary DataFrame from the aggregated data
summary = pd.DataFrame(summary_data)

# Move Value to the third column
cols = summary.columns.tolist()
cols = cols[1:3] + cols[0:1] + cols[3:-1]
summary = summary[cols]

# Create a new table for the total counts
total_counts = summary\
    .set_index(["Model", "Treatment", "Value"])\
    .sum(axis=1)\
    .reset_index(name='Total')

# Display the total counts
print(total_counts)

# Concatenate Model, Treatment, and Value with a hyphen
total_counts['ID'] = total_counts['Model'] + "-" + total_counts['Treatment']

# Create a new table for the final results
final_results = pd.DataFrame()

# For each ID, get the correct, neutral, and incorrect totals
for id, group in total_counts.groupby('ID'):
    correct_total = group[group['Value'] == 1]['Total'].sum()
    neutral_total = group[group['Value'] == 0]['Total'].sum()
    incorrect_total = group[group['Value'] == -1]['Total'].sum()
    print(f"{id}: {correct_total}, {neutral_total}, {incorrect_total}")

    # Add correct and neutral totals
    correct_and_neutral_total = correct_total + neutral_total

    # Compute accuracy
    accuracy = correct_and_neutral_total / (correct_total + neutral_total + incorrect_total)

    # Create a new row for the final results table
    row = {
        'ID': id,
        'Correct': correct_total,
        'Neutral': neutral_total,
        'Incorrect': incorrect_total,
        'Accuracy': accuracy}

    # Add the results to the final results table
    final_results = final_results._append(row, ignore_index=True)

    # Display the accuracy
    print(f"{id}: {accuracy:.4}")
    print()

# Display a bar chart of the final results
final_results.plot.bar(
    x='ID',
    y='Accuracy',
    figsize=(10, 5))
plt.ylim(0.0, 1.0)
plt.title('Accuracy by Model')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=15, ha='right')
plt.subplots_adjust(bottom=0.2)
plt.gca().get_legend().remove()

# Add labels to the bars
for index, row in final_results.iterrows():
    plt.text(
        x=index,
        y=row['Accuracy'] - 0.05,
        s=f"{row['Accuracy']:.3f}",
        ha='center',
        color='white',)

# Save the plot as an SVG file
accuracy_svg_file_path = analysis_folder_path + "/SVG/" + "accuracy-by-model.svg"
plt.savefig(accuracy_svg_file_path, bbox_inches="tight")

# Save the plot as a PNG file
accuracy_png_file_path = analysis_folder_path + "/PNG/" + "accuracy-by-model.png"
plt.savefig(accuracy_png_file_path, bbox_inches="tight", dpi=300)

# Save the data to a CSV file
accuracy_csv_file_path = analysis_folder_path + "/CSV/" + "accuracy-by-model.csv"
final_results.to_csv(accuracy_csv_file_path, index=False)

# Display the plot
plt.show()

# Copy the data
data_to_summarize = data.copy()

# Make zero values positive cases (i.e. correct predictions)
data_to_summarize = data_to_summarize.replace(0, 1)

# Make negative values negative cases (i.e. incorrect predictions)
data_to_summarize = data_to_summarize.replace(-1, 0)

# Remove the case_id column (needed to make aggregation work)
data_to_summarize = data_to_summarize.drop(columns=['case_id'])
data_to_summarize = data_to_summarize.rename(columns={"id": "case_id"})

# Aggregate the details data
grouped_data = data_to_summarize.groupby(['Model', 'Treatment'])
details_data = pd.DataFrame()
for (model, treatment), group in grouped_data:
    numeric_group = group.select_dtypes(include='number')
    row = {'Model': model, 'Treatment': treatment, **numeric_group.mean().to_dict()}
    details_data = details_data._append(row, ignore_index=True)

# For each model-treatment, plot the accuracy of all columns as a separate bar chart
for (model, treatment), group in details_data.groupby(['Model', 'Treatment']):
    group = group.drop(columns=['Model', 'Treatment'])
    group = group.iloc[0].transpose().to_frame()
    group.plot.barh(
        legend=False,
        figsize=(10, 5))
    plt.title(f"{model} - {treatment}")
    plt.xlabel('Accuracy')
    plt.ylabel('Feature')
    plt.gca().invert_yaxis()
    plt.yticks(fontsize=8)
    plt.xticks(rotation=15, ha='right')
    plt.subplots_adjust(left=0.25)

    # Save the plot as an SVG file
    details_svg_file_path = analysis_folder_path + "/SVG/" + f"{model}-{treatment}-details.svg"
    plt.savefig(details_svg_file_path, bbox_inches="tight")

    # Save the plot as a PNG file
    details_png_file_path = analysis_folder_path + "/PNG/" + f"{model}-{treatment}-details.png"
    plt.savefig(details_png_file_path, bbox_inches="tight", dpi=300)

    # Save the data to a CSV file
    details_csv_file_path = analysis_folder_path + "/CSV/" + f"{model}-{treatment}-details.csv"
    final_results.to_csv(details_csv_file_path, index=False)

    # Display the plot
    plt.show()









