# Import libraries
import os
import pandas as pd
import matplotlib.pyplot as plt

# Hide future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Set the file and folder paths
root_folder_path = "C:/Users/Matthew/Dropbox/Personal/School/JHU/Ethics/Project"
input_folder_path = root_folder_path + "/Data/Logs"
output_folder_path = root_folder_path + "/Data/Analysis"

# Get all CSV files in the input folder
input_file_paths = []
for file_name in os.listdir(input_folder_path):
    if file_name.endswith(".csv"):
        file_path = os.path.join(input_folder_path, file_name)
        input_file_paths.append(file_path)

# Create an empty data frame
data = pd.DataFrame()

# Read each input file and add it to the data frame
for input_file_path in input_file_paths:

    # Read the input file
    input_data = pd.read_csv(input_file_path)

    # Get the result ID from the file name
    file_name = os.path.basename(input_file_path)
    file_name = file_name.replace("-baseline", "")
    file_name = file_name.replace(".csv", "")
    file_name = file_name.strip()

    # Insert the result as the first column
    input_data.insert(0, "Results ID", file_name)

    # Add the input data to the output data
    data = data._append(input_data)

# Filter out any error messages
data = data[data["Error Message"].isnull()]

# Filter out rules-explain
data = data[data["Results ID"] != "rules-explain"]

# Capitalize GPT
#data["Results ID"] = data["Results ID"].str.replace("gpt", "GPT")

# Convert to numeric
data["Duration (s)"] = pd.to_numeric(data["Duration (s)"])
data["Total Tokens"] = pd.to_numeric(data["Total Tokens"])

# Compute average duration per token
data["Duration per Token (ms)"] = data["Duration (s)"] / data["Total Tokens"] * 1000

# Group durations by result ID and aggregate
totals = data.groupby("Results ID").agg({
    "Duration (s)": "mean",
    "Duration per Token (ms)": "mean"})

# Reorder the x-axis labels
totals = totals.reindex(["gpt-3.5-explain", "gpt-3.5-verify", "gpt-3.5-correct", "gpt-4-explain", "gpt-4-verify", "gpt-4-correct"])

# Plot the average prediction duration
totals.plot(
    y=["Duration (s)"],
    kind="bar",
    figsize=(10, 5))
plt.figsize=(10, 5)
plt.xticks(rotation=0)
plt.title("Runtime Duration per Inference by Model and Task")
plt.xlabel("Model - Task")
plt.ylabel("Average Duration (s)")
plt.subplots_adjust(
    left=0.1,
    right=0.9)
plt.legend().remove()

# Add a label to each bar
for index, value in enumerate(totals["Duration (s)"]):
    plt.text(
        index,
        value - 2.0,
        str(round(value, 2)),
        ha='center',
        color='white',
        rotation=0)

# Save the plot as an SVG file
runtime_svg_file_path = output_folder_path + "/SVG/" + "runtime-per-inference-by-model.svg"
plt.savefig(runtime_svg_file_path, format="svg", bbox_inches="tight")

# Save the plot as a PNG file
runtime_png_file_path = output_folder_path + "/PNG/" + "runtime-per-inference-by-model.png"
plt.savefig(runtime_png_file_path, format="png", bbox_inches="tight", dpi=300)

# Show the plot
plt.show()

# Save the data as a CSV file
runtime_csv_file_path = output_folder_path + "/CSV/" + "runtime-per-inference-by-model.csv"
totals.to_csv(runtime_csv_file_path)

# Plot the average duration per token
totals.plot(
    y=["Duration per Token (ms)"],
    kind="bar",
    figsize=(10, 5))
plt.xticks(rotation=0)
plt.title("Runtime Duration per Token by Model and Task")
plt.xlabel("Model")
plt.ylabel("Average Duration (ms)")
plt.subplots_adjust(
    left=0.1,
    right=0.9)
plt.legend().remove()

# Add labels to the bars
for index, value in enumerate(totals["Duration per Token (ms)"]):
    plt.text(
        index,
        value - 0.75,
        str(round(value, 2)),
        ha='center',
        color='white',
        rotation=0)

# Save the plot as an SVG file
runtime_per_token_svg_file_path = output_folder_path + "\\SVG\\" + "runtime-per-token-by-model.svg"
plt.savefig(runtime_per_token_svg_file_path, format="svg", bbox_inches="tight")

# Save the plot as a PNG file
runtime_per_token_png_file_path = output_folder_path + "\\PNG\\" + "runtime-per-token-by-model.png"
plt.savefig(runtime_per_token_png_file_path, format="png", bbox_inches="tight", dpi=300)

# Show the plot
plt.show()

# Save the data as a CSV file
runtime_per_token_csv_file_path = output_folder_path + "\\CSV\\" + "runtime-per-token-by-model.csv"
totals.to_csv(runtime_per_token_csv_file_path)
