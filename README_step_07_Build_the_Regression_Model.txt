Step 7 - Build the Regression Model

In this step, I built an ANN regression model to predict diamond prices.

Code file:
step_07_Build the Regression Model.py

Input folder:
step_04_Data Preprocessing and Feature Engineering

Input files:
regression_train.csv
regression_test.csv

Libraries used:
os
pandas
numpy
matplotlib
scikit-learn
tensorflow / keras

Output folder:
step_07_Build the Regression Model

What this step does:
I loaded the prepared regression data from Step 4. The target column is price. I trained an ANN model, made price predictions, and evaluated the model using MAE, MSE, RMSE, and R2.

Main outputs:
- regression_metrics.csv: MAE, MSE, RMSE, and R2 results
- regression_predictions.csv: actual price, predicted price, and prediction error
- training_history.csv: training and validation history
- regression_model_structure.txt: model layers and parameters
- training_validation_loss.png: training and validation loss plot
- training_validation_mae.png: training and validation MAE plot
- actual_vs_predicted_price_scatter.png: actual price vs predicted price with ideal line
- actual_vs_predicted_price_comparison.png: first 200 actual and predicted prices
- prediction_error_distribution.png: distribution of prediction errors
- regression_metrics_plot.png: summary of main regression metrics
- price_ann_model.keras: saved trained regression model
- regression_model_report.txt: summary report for this step

Why this step matters:
This step shows how I used an ANN model to estimate diamond price and checked how close the predicted prices were to the real prices.