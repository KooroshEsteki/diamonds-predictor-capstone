# Step 3 - Exploratory Data Analysis (EDA)
# In this step, I explore the Diamonds dataset.
# I check missing values, duplicates, descriptive statistics,
# feature distributions, relationships between features, and patterns.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Dataset path
file_path = "diamonds.csv"


# Output folder for this step
output_folder = "step_03_Exploratory Data Analysis (EDA)"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# A simple text file to save the main printed results
report_path = os.path.join(output_folder, "eda_report.txt")
report = []


# Load the dataset
df = pd.read_csv(file_path)


# Remove extra index column if it exists
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])


report.append("Step 3 - Exploratory Data Analysis (EDA)\n")
report.append("Dataset loaded successfully.\n")
report.append("\nDataset shape:\n")
report.append(str(df.shape) + "\n")


#
# 1. Missing values and duplicate rows
# 

missing_values = df.isnull().sum()
missing_values.to_csv(os.path.join(output_folder, "missing_values.csv"))

duplicate_count = df.duplicated().sum()

report.append("\nMissing values in each column:\n")
report.append(str(missing_values) + "\n")

report.append("\nNumber of duplicate rows:\n")
report.append(str(duplicate_count) + "\n")



# 2. Descriptive statistics


numeric_summary = df.describe()
numeric_summary.to_csv(os.path.join(output_folder, "numeric_summary.csv"))

report.append("\nDescriptive statistics for numerical columns:\n")
report.append(str(numeric_summary) + "\n")


categorical_columns = df.columns[df.dtypes == "object"]

report.append("\nSummary for categorical columns:\n")

if len(categorical_columns) > 0:
    categorical_summary = df[categorical_columns].describe()
    categorical_summary.to_csv(os.path.join(output_folder, "categorical_summary.csv"))
    report.append(str(categorical_summary) + "\n")
else:
    report.append("There are no categorical columns.\n")



# 3. Feature distributions


numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns

for column in numeric_columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[column], kde=True)
    plt.title("Distribution of " + column)
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, column + "_distribution.png"))
    plt.close()


for column in categorical_columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(x=column, data=df)
    plt.title("Count of " + column)
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, column + "_count.png"))
    plt.close()


report.append("\nFeature distribution plots were saved.\n")



# 4. Correlation matrix


correlation_matrix = df[numeric_columns].corr()
correlation_matrix.to_csv(os.path.join(output_folder, "correlation_matrix.csv"))

plt.figure(figsize=(9, 7))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "correlation_matrix.png"))
plt.close()

report.append("\nCorrelation matrix:\n")
report.append(str(correlation_matrix) + "\n")



# 5. Feature relationships


plt.figure(figsize=(8, 5))
sns.scatterplot(x="carat", y="price", data=df)
plt.title("Carat vs Price")
plt.xlabel("Carat")
plt.ylabel("Price")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "carat_vs_price.png"))
plt.close()


plt.figure(figsize=(8, 5))
sns.boxplot(x="cut", y="price", data=df)
plt.title("Price by Cut")
plt.xlabel("Cut")
plt.ylabel("Price")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "price_by_cut.png"))
plt.close()


plt.figure(figsize=(8, 5))
sns.boxplot(x="color", y="price", data=df)
plt.title("Price by Color")
plt.xlabel("Color")
plt.ylabel("Price")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "price_by_color.png"))
plt.close()


plt.figure(figsize=(9, 5))
sns.boxplot(x="clarity", y="price", data=df)
plt.title("Price by Clarity")
plt.xlabel("Clarity")
plt.ylabel("Price")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "price_by_clarity.png"))
plt.close()


report.append("\nRelationship plots were saved.\n")



# 6. Trends and patterns


avg_price_cut = df.groupby("cut")["price"].mean().sort_values(ascending=False)
avg_price_cut.to_csv(os.path.join(output_folder, "average_price_by_cut.csv"))

avg_price_color = df.groupby("color")["price"].mean().sort_values(ascending=False)
avg_price_color.to_csv(os.path.join(output_folder, "average_price_by_color.csv"))

avg_price_clarity = df.groupby("clarity")["price"].mean().sort_values(ascending=False)
avg_price_clarity.to_csv(os.path.join(output_folder, "average_price_by_clarity.csv"))


report.append("\nAverage price by cut:\n")
report.append(str(avg_price_cut) + "\n")

report.append("\nAverage price by color:\n")
report.append(str(avg_price_color) + "\n")

report.append("\nAverage price by clarity:\n")
report.append(str(avg_price_clarity) + "\n")


# Check rows where diamond dimensions are zero
zero_dimensions = df[(df["x"] == 0) | (df["y"] == 0) | (df["z"] == 0)]
zero_dimensions.to_csv(os.path.join(output_folder, "rows_with_zero_dimensions.csv"), index=False)

report.append("\nRows with zero values in x, y, or z:\n")
report.append(str(zero_dimensions.shape[0]) + "\n")


# Save the text report
with open(report_path, "w", encoding="utf-8") as file:
    file.writelines(report)


print("Step 3 - EDA is complete.")
print("All results were saved inside this folder:")
print(output_folder)