# Step 2 - Data Collection
# In this step, I load the Diamonds dataset using Pandas.
# I also inspect the dataset structure, columns, data types, and basic information.

import pandas as pd
import os


# This is the path to my Diamonds dataset
file_path = "diamonds.csv"


# Check if the dataset file exists
if not os.path.exists(file_path):
    print("The dataset file was not found.")
    print("Please make sure diamonds.csv is inside data/raw/")
else:
    print("The dataset file was found.")

    # Load the dataset
    df = pd.read_csv(file_path)

    # Some versions of the Kaggle dataset have an extra index column
    # I remove it because it is not useful for the analysis
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    print("\nDataset loaded successfully.")

    # Show the first 5 rows
    print("\nFirst 5 rows of the dataset:")
    print(df.head())

    # Show the number of rows and columns
    print("\nDataset shape:")
    print(df.shape)

    # Show the column names
    print("\nColumn names:")
    print(df.columns.tolist())

    # Show column information and data types
    print("\nDataset information:")
    print(df.info())

    # Show data types separately
    print("\nData types:")
    print(df.dtypes)

    # Check missing values
    print("\nMissing values in each column:")
    print(df.isnull().sum())

    # Check duplicate rows
    print("\nNumber of duplicate rows:")
    print(df.duplicated().sum())

    # Show basic statistics for numerical columns
    print("\nBasic statistics for numerical columns:")
    print(df.describe())

    # Show basic information for categorical columns
    print("\nBasic information for categorical columns:")

    categorical_columns = df.columns[df.dtypes == "object"]

    if len(categorical_columns) > 0:
        print(df[categorical_columns].describe())
    else:
        print("There are no categorical columns in this dataset.")

    print("\nStep 2 - Data Collection is complete.")