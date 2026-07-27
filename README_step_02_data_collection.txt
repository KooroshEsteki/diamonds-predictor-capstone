Step 2 - Data Collection

In this step, I loaded the Diamonds dataset into Python and checked the basic structure of the data.

Code file name:
step_02_data_collection.py

Dataset file name:
diamonds.csv

Dataset location:
The dataset file diamonds.csv should be in the same directory as the Step 2 Python code.

Libraries used:
pandas
os

What this step reads:
This step reads the Diamonds dataset from:

diamonds.csv

The code uses Pandas to load the dataset into a dataframe. It also checks whether the file exists before trying to read it.

What this step does:
In this step, I checked the dataset before moving to the analysis and machine learning parts. I inspected the first few rows, column names, dataset shape, data types, missing values, duplicate rows, numerical statistics, and categorical column summaries.

Outputs from this step:
This step does not save a new CSV or plot file. The output is printed in the Python console.

The console output includes:

1. File check
The code checks whether diamonds.csv exists in the same folder.

2. First 5 rows
The code prints the first 5 rows of the dataset so I can quickly see what the data looks like.

3. Dataset shape
The code prints the number of rows and columns in the dataset.

4. Column names
The code prints all column names, such as carat, cut, color, clarity, depth, table, price, x, y, and z.

5. Dataset information
The code prints information about the dataset columns and data types.

6. Data types
The code prints the type of each column separately.

7. Missing values
The code checks how many missing values exist in each column.

8. Duplicate rows
The code checks how many duplicate rows are in the dataset.

9. Numerical statistics
The code prints basic statistics for numerical columns, such as mean, standard deviation, minimum, maximum, and quartiles.

10. Categorical summary
The code prints a summary for categorical columns such as cut, color, and clarity.

Why this step is important:
This step helps me understand the dataset before doing exploratory data analysis, preprocessing, feature engineering, and machine learning. It also helps me check whether the dataset loaded correctly and whether there are any obvious issues such as missing values or duplicate rows.

Notes:
Some versions of the Kaggle Diamonds dataset include an extra column called Unnamed: 0. In my code, I remove this column because it is only an index column and is not useful for the analysis.

Step 2 is complete when the dataset loads successfully and the console shows the dataset structure, data types, missing values, duplicates, and basic statistics.