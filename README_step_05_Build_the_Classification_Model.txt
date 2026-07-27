Step 5 - Build the Classification Model

In this step, I built an ANN model to predict diamond clarity.

Code file:
step_05_Build the Classification Model.py

Input folder:
step_04_Data Preprocessing and Feature Engineering

Input files:
classification_train.csv
classification_test.csv
clarity_label_mapping.csv

Libraries used:
os
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow / keras

Output folder:
step_05_Build the Classification Model

Main outputs:
- classification_metrics.csv: accuracy, precision, recall, and F1-score
- classification_report.txt: detailed results for each clarity class
- confusion_matrix.csv: actual vs predicted clarity counts
- classification_predictions.csv: actual and predicted clarity values
- training_history.csv: training and validation history
- training_validation_accuracy.png: training and validation accuracy plot
- training_validation_loss.png: training and validation loss plot
- classification_metrics_plot.png: metric summary plot
- confusion_matrix_plot.png: confusion matrix plot
- actual_vs_predicted_clarity_distribution.png: actual vs predicted clarity distribution
- clarity_ann_model.keras: saved trained ANN model
- classification_model_report.txt: short summary report

Main result:
The model reached about 62% accuracy. This means it learned useful patterns, but it still confused some similar clarity classes.

Why this step matters:
This step shows how I used an ANN to classify diamond clarity and evaluated the model using common classification metrics.