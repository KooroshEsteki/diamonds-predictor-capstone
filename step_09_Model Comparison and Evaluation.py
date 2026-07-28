# Step 9 - Model Comparison and Evaluation
# In this step, I compare the models and approaches used in the project.
# I compare the classification model, regression model, and clustering approach.
# Since these models solve different tasks, I compare them by purpose,
# metrics, strengths, limitations, and possible improvements.

import pandas as pd
import os


# 1. Folder paths

classification_folder = "step_05_Build the Classification Model"
regression_folder = "step_07_Build the Regression Model"
clustering_folder = "step_08_Perform Customer Segmentation"

output_folder = "step_09_Model Comparison and Evaluation"

if not os.path.exists(output_folder):os.makedirs(output_folder)


# Clean old Step 9 output files
# I only want to keep the new report and summary csv for this step.

for file_name in os.listdir(output_folder):
    file_path = os.path.join(output_folder, file_name)

    if os.path.isfile(file_path):
        os.remove(file_path)


# 2. File paths from previous steps

classification_metrics_file = os.path.join(
    classification_folder, "classification_metrics.csv")

regression_metrics_file = os.path.join(
    regression_folder,"regression_metrics.csv")

cluster_summary_file = os.path.join(
    clustering_folder,"cluster_summary.csv")


# 3. Load previous results

classification_metrics = pd.read_csv(classification_metrics_file)
regression_metrics = pd.read_csv(regression_metrics_file)
cluster_summary = pd.read_csv(cluster_summary_file)

print("Previous model results loaded successfully.")


# 4. Clean cluster summary column name if needed

if "Unnamed: 0" in cluster_summary.columns:
    cluster_summary = cluster_summary.rename(columns={"Unnamed: 0": "cluster"})


# 5. Get important values from the previous steps

classification_accuracy = classification_metrics.loc[
    classification_metrics["Metric"] == "Accuracy","Score"].values[0]

classification_precision = classification_metrics.loc[
    classification_metrics["Metric"] == "Precision","Score"].values[0]

classification_recall = classification_metrics.loc[
    classification_metrics["Metric"] == "Recall","Score"].values[0]

classification_f1 = classification_metrics.loc[
    classification_metrics["Metric"] == "F1-score","Score"].values[0]


regression_mae = regression_metrics.loc[
    regression_metrics["Metric"] == "MAE","Score"].values[0]

regression_mse = regression_metrics.loc[
    regression_metrics["Metric"] == "MSE","Score"].values[0]

regression_rmse = regression_metrics.loc[
    regression_metrics["Metric"] == "RMSE","Score"].values[0]

regression_r2 = regression_metrics.loc[
    regression_metrics["Metric"] == "R2","Score"].values[0]


number_of_clusters = cluster_summary.shape[0]

total_clustered_diamonds = cluster_summary["number_of_diamonds"].sum()

highest_price_cluster = cluster_summary.loc[
    cluster_summary["price"].idxmax(),"cluster"]

lowest_price_cluster = cluster_summary.loc[
    cluster_summary["price"].idxmin(),"cluster"]


# 6. Create model comparison summary table

comparison_summary = pd.DataFrame({
    "Model_or_Approach": [
        "Classification ANN",
        "Regression ANN",
        "K-Means Clustering"],
    "Main_Task": [
        "Classify diamond clarity",
        "Predict diamond price",
        "Segment diamonds into groups"],
    "Target": [
        "clarity",
        "price",
        "No target column"],
    "Main_Metrics": [
        "Accuracy, Precision, Recall, F1-score",
        "MAE, MSE, RMSE, R2",
        "Inertia, elbow method, cluster summary"],
    "Main_Result": [
        "Accuracy = " + str(round(classification_accuracy, 4)) +
        ", F1-score = " + str(round(classification_f1, 4)),

        "MAE = " + str(round(regression_mae, 2)) +
        ", RMSE = " + str(round(regression_rmse, 2)) +
        ", R2 = " + str(round(regression_r2, 4)),

        str(number_of_clusters) + " clusters were used"],
    "Strength": [
        "Useful for predicting diamond clarity categories",
        "Useful for estimating diamond prices",
        "Useful for finding diamond groups or market segments"],
    "Limitation": [
        "Some clarity categories are similar and can be confused",
        "Prediction errors can be larger for expensive diamonds",
        "There is no true accuracy score because clustering is unsupervised"],
    "Possible_Improvement": [
        "Tune ANN structure, use class balancing, or test other classification models",
        "Tune ANN structure, use more feature engineering, or test other regression models",
        "Try different numbers of clusters or use extra clustering quality metrics"]})

comparison_summary.to_csv(
    os.path.join(output_folder, "model_comparison_summary.csv"),
    index=False)


# 7. Save final report in my own words

report = []

report.append("Step 9 - Model Comparison and Evaluation\n")

report.append("\nIn this step, I compared the three main approaches used in this project.\n")
report.append("These approaches are the classification ANN, regression ANN, and K-Means clustering.\n")
report.append("They cannot be compared using one single score because they solve different problems.\n")

report.append("\nClassification model:\n")
report.append("The classification model was used to predict diamond clarity.\n")
report.append("Accuracy: " + str(classification_accuracy) + "\n")
report.append("Precision: " + str(classification_precision) + "\n")
report.append("Recall: " + str(classification_recall) + "\n")
report.append("F1-score: " + str(classification_f1) + "\n")
report.append("This model is useful for predicting clarity categories, but some clarity groups are close to each other and may be difficult to separate.\n")

report.append("\nRegression model:\n")
report.append("The regression model was used to predict diamond prices.\n")
report.append("MAE: " + str(regression_mae) + "\n")
report.append("MSE: " + str(regression_mse) + "\n")
report.append("RMSE: " + str(regression_rmse) + "\n")
report.append("R2: " + str(regression_r2) + "\n")
report.append("This model is useful for price estimation. MAE and RMSE show the average prediction error in dollars, while R2 shows how much of the price variation is explained by the model.\n")

report.append("\nClustering approach:\n")
report.append("The clustering approach was used to segment diamonds into groups.\n")
report.append("Number of clusters used: " + str(number_of_clusters) + "\n")
report.append("Total clustered diamonds: " + str(int(total_clustered_diamonds)) + "\n")
report.append("Highest average price cluster: " + str(highest_price_cluster) + "\n")
report.append("Lowest average price cluster: " + str(lowest_price_cluster) + "\n")
report.append("Since clustering is unsupervised, there is no true accuracy score. Instead, I used the elbow method and cluster summaries to understand the clustering result.\n")

report.append("\nOverall comparison:\n")
report.append("The classification model is best for predicting diamond clarity categories.\n")
report.append("The regression model is best for estimating diamond price.\n")
report.append("The clustering model is best for grouping diamonds into market-like segments.\n")
report.append("Because each model has a different purpose, I compared them based on task, metrics, strengths, and limitations instead of directly ranking them by one score.\n")

report.append("\nPossible improvements:\n")
report.append("For the classification model, I could tune the ANN structure, use class balancing, or compare with other classification models.\n")
report.append("For the regression model, I could add more feature engineering, tune the ANN, or compare with models such as Random Forest or XGBoost.\n")
report.append("For clustering, I could test different numbers of clusters and add more clustering evaluation metrics such as silhouette score.\n")

report.append("\nSaved output files:\n")
report.append("model_comparison_summary.csv\n")
report.append("model_comparison_report.txt\n")

report.append("\nStep 9 is complete.\n")

with open(
    os.path.join(output_folder, "model_comparison_report.txt"),
    "w",
    encoding="utf-8") as file:
    file.writelines(report)


# 8. Final message

print("\nStep 9 - Model Comparison and Evaluation is complete.")
print("Only these files were saved inside this folder:")
print("model_comparison_summary.csv")
print("model_comparison_report.txt")