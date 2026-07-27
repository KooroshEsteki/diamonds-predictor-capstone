Step 4 - Data Preprocessing and Feature Engineering

In this step, I prepared the Diamonds dataset for the machine learning parts of the project.

Code file:
step_04_Data Preprocessing and Feature Engineering.py

Input file:
diamonds.csv

Input location:
diamonds.csv should be in the same folder as the code.

Libraries used:
pandas
os
scikit-learn

Output folder:
step_04_Data Preprocessing and Feature Engineering

What this step does:
I loaded the dataset, checked missing values, removed rows with zero values in x, y, or z, created a new volume feature, encoded categorical columns, scaled numerical features, and prepared separate files for classification, regression, and clustering.

Main outputs:
- feature_engineered_dataset.csv: cleaned dataset with the new volume column
- encoded_dataset.csv: dataset with categorical variables encoded
- clarity_label_mapping.csv: mapping between clarity labels and numbers
- classification_train.csv: training data for clarity classification
- classification_test.csv: testing data for clarity classification
- classification_features.csv: feature names used for classification
- regression_train.csv: training data for price prediction
- regression_test.csv: testing data for price prediction
- regression_features.csv: feature names used for regression
- clustering_dataset.csv: scaled dataset for clustering
- clustering_features.csv: feature names used for clustering
- preprocessing_report.txt: summary of the preprocessing results

Main result:
The original dataset had 53,940 rows and 10 columns.
After removing 20 rows with zero dimensions, the final dataset had 53,920 rows and 11 columns.
The new column added was volume = x * y * z.

Why this step matters:
This step prepares clean and numerical data so it can be used later for classification, regression, and clustering models.