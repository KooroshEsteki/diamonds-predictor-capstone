Step 9 - Model Comparison and Evaluation

In this step, I compared the main models used in my project.

Code file:
step_09_Model Comparison and Evaluation.py

Input folders:
step_05_Build the Classification Model
step_07_Build the Regression Model
step_08_Perform Customer Segmentation

Input files:
classification_metrics.csv
regression_metrics.csv
cluster_summary.csv

Libraries used:
pandas
os

Output folder:
step_09_Model Comparison and Evaluation

What this step does:
I collected the results from the classification model, regression model, and clustering step. Since these models solve different tasks, I did not compare them using one single score. Instead, I compared their purpose, metrics, strengths, limitations, and possible improvements.

Main outputs:
- model_comparison_summary.csv: comparison table for all models
- model_comparison_report.txt: written summary of the comparison

Why this step matters:
This step helps me summarize what each model was used for, how well it performed, and what could be improved later.