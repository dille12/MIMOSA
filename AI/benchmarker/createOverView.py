import os
import pandas as pd
import json
# Define the main directory
results_dir = "AI/benchmarker/results"

# Initialize a list to store DataFrames
dataframes = []

data = {}

# Loop through each folder in the results directory
for folder in os.listdir(results_dir):
    folder_path = os.path.join(results_dir, folder)
    if "ITER" not in folder_path:
        continue

    if "ITER17_RETRAIN_1262IMAGES_SHAPE_256x256x1_1.keras_20250331_123919" not in folder_path:
        continue

    data[folder_path] = {"dataframe": [], "robustness": []}

    ROBUSTNESS = {}
    # Check if it's a directory
    if os.path.isdir(folder_path):
        # Look for CSV files in the folder
        for file in os.listdir(folder_path):
            if file.endswith("summary_results.csv"):
                file_path = os.path.join(folder_path, file)
                print(file_path)
                try:
                    df = pd.read_csv(file_path, delimiter=",")  # Read the CSV file
                    df['source_folder'] = folder  # Add a column to track the source folder
                    dataframes.append(df)
                    data[folder_path]["dataframe"] = df
                except:
                    print("Couldnt read file")
            
            if "robustness_results" in file:
                file_path = os.path.join(folder_path, file)
                print(file_path)
                with open(file_path, "r") as js:
                    d = json.load(js)  # Load JSON into a Python dictionary
                ROBUSTNESS[file] = d
    

    ROBUSTNESS["overview"] = {}

    for x in ROBUSTNESS:
        if "tests" not in ROBUSTNESS[x]:
            continue
        for y in ROBUSTNESS[x]["tests"]:
            for z in ROBUSTNESS[x]["tests"][y]:
                if z not in ROBUSTNESS["overview"]:
                    ROBUSTNESS["overview"][z] = []
                ROBUSTNESS["overview"][z].append(ROBUSTNESS[x]["tests"][y][z])
                print(x, y, z)

    
    data[folder_path]["robustness"] = ROBUSTNESS


    print(ROBUSTNESS)
    print(df)

    


