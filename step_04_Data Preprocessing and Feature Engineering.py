# Step 4 - Data Preprocessing and Feature Engineering
# In this step, I prepare the Diamonds dataset for machine learning.
# I handle missing values, encode categorical variables, scale numerical features,
# select relevant features, and split the dataset into training and testing sets.
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
# 1. File paths and output folder
file_path = "diamonds.csv"
output_folder = "step_04_Data Preprocessing and Feature Engineering"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
report = []

# 2. Load the dataset

df = pd.read_csv(file_path)
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])
print("Dataset loaded successfully.")
print("Original dataset shape:", df.shape)
report.append("Step 4 - Data Preprocessing and Feature Engineering\n")
report.append("\nOriginal dataset shape:\n")
report.append(str(df.shape) + "\n")

# 3. Handle missing values

missing_before = df.isnull().sum()
report.append("\nMissing values before preprocessing:\n")
report.append(str(missing_before) + "\n")
numerical_columns = df.select_dtypes(include=["number"]).columns
categorical_columns = df.columns[df.dtypes == "object"]
for column in numerical_columns:
    df[column] = df[column].fillna(df[column].median())
for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])
missing_after = df.isnull().sum()
report.append("\nMissing values after preprocessing:\n")
report.append(str(missing_after) + "\n")

# 4. Remove invalid dimension values
# The x, y, and z columns are diamond dimensions.
# They should not be zero, so I remove rows where any of them are zero.

rows_before_cleaning = df.shape[0]
df = df[(df["x"] > 0) & (df["y"] > 0) & (df["z"] > 0)]
rows_after_cleaning = df.shape[0]
removed_rows = rows_before_cleaning - rows_after_cleaning
report.append("\nRows removed because of zero x, y, or z values:\n")
report.append(str(removed_rows) + "\n")

# 5. Feature engineering
# I create a new feature called volume using the diamond dimensions.

df["volume"] = df["x"] * df["y"] * df["z"]
df.to_csv(os.path.join(output_folder, "feature_engineered_dataset.csv"), index=False)
report.append("\nNew feature created:\n")
report.append("volume = x * y * z\n")

# 6. Encode categorical variables
# I save a fully encoded version of the dataset.
# This is useful because machine learning models need numerical values.

df_encoded = pd.get_dummies(df, columns=["cut", "color", "clarity"], drop_first=False)
df_encoded.to_csv(os.path.join(output_folder, "encoded_dataset.csv"), index=False)
report.append("\nCategorical variables encoded:\n")
report.append("cut, color, clarity\n")

# 7. Prepare data for clarity classification
# For classification, the target is clarity.
# I encode clarity into numbers and use the other columns as features.

classification_df = df.copy()
clarity_encoder = LabelEncoder()
classification_df["clarity_encoded"] = clarity_encoder.fit_transform(classification_df["clarity"])
clarity_mapping = pd.DataFrame({"clarity": clarity_encoder.classes_, "clarity_encoded": range(len(clarity_encoder.classes_))})
clarity_mapping.to_csv(os.path.join(output_folder, "clarity_label_mapping.csv"), index=False)
X_classification = classification_df.drop(columns=["clarity", "clarity_encoded"])
y_classification = classification_df["clarity_encoded"]
X_classification = pd.get_dummies(X_classification, columns=["cut", "color"], drop_first=False)
X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(X_classification, y_classification, test_size=0.2, random_state=42)
scaler_classification = StandardScaler()
X_class_train_scaled = scaler_classification.fit_transform(X_class_train)
X_class_test_scaled = scaler_classification.transform(X_class_test)
X_class_train_scaled = pd.DataFrame(X_class_train_scaled, columns=X_class_train.columns)
X_class_test_scaled = pd.DataFrame(X_class_test_scaled, columns=X_class_test.columns)
classification_train = X_class_train_scaled.copy()
classification_train["clarity_encoded"] = y_class_train.values
classification_test = X_class_test_scaled.copy()
classification_test["clarity_encoded"] = y_class_test.values
classification_train.to_csv(os.path.join(output_folder, "classification_train.csv"), index=False)
classification_test.to_csv(os.path.join(output_folder, "classification_test.csv"), index=False)
classification_features = pd.DataFrame({"classification_features": X_classification.columns})
classification_features.to_csv(os.path.join(output_folder, "classification_features.csv"), index=False)
report.append("\nClassification data prepared:\n")
report.append("Target column: clarity_encoded\n")
report.append("Training file: classification_train.csv\n")
report.append("Testing file: classification_test.csv\n")

# 8. Prepare data for price regression
# For regression, the target is price.
# I use the other diamond features to predict price later.

regression_df = df.copy()
X_regression = regression_df.drop(columns=["price"])
y_regression = regression_df["price"]
X_regression = pd.get_dummies(X_regression, columns=["cut", "color", "clarity"], drop_first=False)
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_regression, y_regression, test_size=0.2, random_state=42)
scaler_regression = StandardScaler()
X_reg_train_scaled = scaler_regression.fit_transform(X_reg_train)
X_reg_test_scaled = scaler_regression.transform(X_reg_test)
X_reg_train_scaled = pd.DataFrame(X_reg_train_scaled, columns=X_reg_train.columns)
X_reg_test_scaled = pd.DataFrame(X_reg_test_scaled, columns=X_reg_test.columns)
regression_train = X_reg_train_scaled.copy()
regression_train["price"] = y_reg_train.values
regression_test = X_reg_test_scaled.copy()
regression_test["price"] = y_reg_test.values
regression_train.to_csv(os.path.join(output_folder, "regression_train.csv"), index=False)
regression_test.to_csv(os.path.join(output_folder, "regression_test.csv"), index=False)
regression_features = pd.DataFrame({"regression_features": X_regression.columns})
regression_features.to_csv(os.path.join(output_folder, "regression_features.csv"), index=False)
report.append("\nRegression data prepared:\n")
report.append("Target column: price\n")
report.append("Training file: regression_train.csv\n")
report.append("Testing file: regression_test.csv\n")

# 9. Prepare data for clustering
# For clustering, there is no target column.
# I encode and scale the full dataset so it can be used for clustering later.

clustering_df = df.copy()
clustering_df = pd.get_dummies(clustering_df, columns=["cut", "color", "clarity"], drop_first=False)
scaler_clustering = StandardScaler()
clustering_scaled = scaler_clustering.fit_transform(clustering_df)
#After scaling, the data becomes a NumPy array.
#This line converts it back into a Pandas DataFrame and keeps the column names.
clustering_scaled = pd.DataFrame(clustering_scaled, columns=clustering_df.columns)
clustering_scaled.to_csv(os.path.join(output_folder, "clustering_dataset.csv"), index=False)
clustering_features = pd.DataFrame({"clustering_features": clustering_df.columns})
clustering_features.to_csv(os.path.join(output_folder, "clustering_features.csv"), index=False)
report.append("\nClustering data prepared:\n")
report.append("There is no target column for clustering.\n")
report.append("Output file: clustering_dataset.csv\n")

# 10. Save preprocessing report

report.append("\nFinal dataset shape after preprocessing:\n")
report.append(str(df.shape) + "\n")
report.append("\nScaling method used:\n")
report.append("StandardScaler\n")
report.append("\nSaved output files:\n")
report.append("feature_engineered_dataset.csv\n")
report.append("encoded_dataset.csv\n")
report.append("clarity_label_mapping.csv\n")
report.append("classification_train.csv\n")
report.append("classification_test.csv\n")
report.append("classification_features.csv\n")
report.append("regression_train.csv\n")
report.append("regression_test.csv\n")
report.append("regression_features.csv\n")
report.append("clustering_dataset.csv\n")
report.append("clustering_features.csv\n")
report.append("preprocessing_report.txt\n")
report.append("\nStep 4 is complete.\n")
with open(os.path.join(output_folder, "preprocessing_report.txt"), "w", encoding="utf-8") as file:
    file.writelines(report)
    
# 11. Final message

print("Step 4 - Data Preprocessing and Feature Engineering is complete.")
print("All output files were saved inside this folder:")
print(output_folder)