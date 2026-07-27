# Step 7 - Build the Regression Model
# In this step, I build an Artificial Neural Network (ANN)
# to predict diamond prices.
# I train and validate the regression model.
# Then I evaluate the model using MAE, MSE, RMSE, and R2.

import os

# These lines reduce TensorFlow warning messages
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping


# 1. Folder paths

input_folder="step_04_Data Preprocessing and Feature Engineering"
output_folder="step_07_Build the Regression Model"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

train_file=os.path.join(input_folder,"regression_train.csv")
test_file=os.path.join(input_folder,"regression_test.csv")


# 2. Load the prepared regression data

train_data=pd.read_csv(train_file)
test_data=pd.read_csv(test_file)

print("Training and testing data loaded successfully.")
print("Training data shape:",train_data.shape)
print("Testing data shape:",test_data.shape)


# 3. Split features and target

target_column="price"

X_train=train_data.drop(columns=[target_column])
y_train=train_data[target_column]

X_test=test_data.drop(columns=[target_column])
y_test=test_data[target_column]

number_of_features=X_train.shape[1]

print("Number of features:",number_of_features)


# 4. Scale the target price

# The feature columns were already scaled in Step 4.
# I scale the price target here because ANN models usually train better
# when the target values are not very large.

price_scaler=StandardScaler()

y_train_scaled=price_scaler.fit_transform(y_train.values.reshape(-1,1))
y_test_scaled=price_scaler.transform(y_test.values.reshape(-1,1))


# 5. Build the ANN regression model

np.random.seed(42)
tf.random.set_seed(42)

model=Sequential()

model.add(Input(shape=(number_of_features,)))
model.add(Dense(128,activation="relu"))
model.add(Dense(64,activation="relu"))
model.add(Dense(32,activation="relu"))
model.add(Dense(1))

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"])

print("\nModel summary:")
model.summary()


# Save model structure in a text file

model_summary=[]

model.summary(print_fn=lambda x: model_summary.append(x))

with open(
    os.path.join(output_folder,"regression_model_structure.txt"),
    "w",
    encoding="utf-8") as file:
    file.write("\n".join(model_summary))


# 6. Train and validate the model

early_stop=EarlyStopping(
    monitor="val_loss",
    patience=30,
    restore_best_weights=True)

history=model.fit(
    X_train,
    y_train_scaled,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1)


# 7. Make predictions

y_pred_scaled=model.predict(X_test)

y_pred=price_scaler.inverse_transform(y_pred_scaled)
y_pred=y_pred.flatten()


# 8. Evaluate the model

mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_pred)

print("\nRegression Model Evaluation Results")
print("MAE:",mae)
print("MSE:",mse)
print("RMSE:",rmse)
print("R2:",r2)


# 9. Save evaluation results

metrics_table=pd.DataFrame({
    "Metric":["MAE","MSE","RMSE","R2"],
    "Score":[mae,mse,rmse,r2]})

metrics_table.to_csv(
    os.path.join(output_folder,"regression_metrics.csv"),
    index=False)


# 10. Save predictions

predictions=pd.DataFrame({
    "actual_price":y_test.values,
    "predicted_price":y_pred})

predictions["prediction_error"]=predictions["actual_price"]-predictions["predicted_price"]

predictions.to_csv(
    os.path.join(output_folder,"regression_predictions.csv"),
    index=False)


# 11. Save training history

history_table=pd.DataFrame(history.history)

history_table.to_csv(
    os.path.join(output_folder,"training_history.csv"),
    index=False)


# 12. Save useful plots

# Training and validation loss

plt.figure(figsize=(8,5))
plt.plot(history_table["loss"],label="Training loss",color="blue")
plt.plot(history_table["val_loss"],label="Validation loss",color="orange")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_folder,"training_validation_loss.png"))
plt.close()


# Training and validation MAE

plt.figure(figsize=(8,5))
plt.plot(history_table["mae"],label="Training MAE",color="blue")
plt.plot(history_table["val_mae"],label="Validation MAE",color="orange")
plt.title("Training and Validation MAE")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_folder,"training_validation_mae.png"))
plt.close()


# Actual price vs predicted price scatter plot with ideal line

plt.figure(figsize=(7,6))
plt.scatter(
    predictions["actual_price"],
    predictions["predicted_price"],
    color="blue",
    alpha=0.6,
    edgecolors="black",
    s=35)

max_value=max(
    predictions["actual_price"].max(),
    predictions["predicted_price"].max())

plt.plot(
    [0,max_value],
    [0,max_value],
    color="red",
    linestyle="--",
    linewidth=2,
    label="Ideal line")

plt.title("Actual Price vs Predicted Price")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_folder,"actual_vs_predicted_price_scatter.png"))
plt.close()


# Compare actual and predicted prices for the first 200 test samples
# This plot uses both scatter points and lines.

comparison_data=predictions.head(200).copy()
comparison_data=comparison_data.reset_index(drop=True)

plt.figure(figsize=(14,6))

# Actual price line and markers
plt.plot(
    comparison_data.index,
    comparison_data["actual_price"],
    color="blue",
    linestyle="-",
    linewidth=1,
    alpha=0.7,
    label="Actual Price")

plt.scatter(
    comparison_data.index,
    comparison_data["actual_price"],
    color="blue",
    marker="o",
    alpha=0.8,
    s=25)

# Predicted price line and markers
plt.plot(
    comparison_data.index,
    comparison_data["predicted_price"],
    color="red",
    linestyle="--",
    linewidth=1,
    alpha=0.7,
    label="Predicted Price")

plt.scatter(
    comparison_data.index,
    comparison_data["predicted_price"],
    color="red",
    marker="x",
    alpha=0.9,
    s=25)

plt.title("Actual vs Predicted Price for First 200 Test Samples")
plt.xlabel("Sample Number")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_folder,"actual_vs_predicted_price_comparison.png"))
plt.close()


# Prediction error distribution

plt.figure(figsize=(8,5))
plt.hist(
    predictions["prediction_error"],
    bins=30,
    color="purple",
    edgecolor="black")

plt.title("Prediction Error Distribution")
plt.xlabel("Actual Price - Predicted Price")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(output_folder,"prediction_error_distribution.png"))
plt.close()


# MAE, RMSE, and R2 plot
# I do not include MSE here because it is much larger than the other values.

metrics_for_plot=pd.DataFrame({
    "Metric":["MAE","RMSE","R2"],
    "Score":[mae,rmse,r2]})

plt.figure(figsize=(8,5))
plt.bar(
    metrics_for_plot["Metric"],
    metrics_for_plot["Score"],
    color=["blue","orange","green"])

plt.title("Regression Model Metrics")
plt.xlabel("Metric")
plt.ylabel("Score")
plt.tight_layout()
plt.savefig(os.path.join(output_folder,"regression_metrics_plot.png"))
plt.close()


# 13. Save the trained model

model.save(os.path.join(output_folder,"price_ann_model.keras"))


# 14. Save a simple report in my own words

report=[]

report.append("Step 7 - Build the Regression Model\n")

report.append("\nIn this step, I built an Artificial Neural Network model to predict diamond prices.\n")
report.append("I used the prepared regression training and testing files from Step 4.\n")
report.append("The target column for this model is price.\n")

report.append("\nTraining data shape:\n")
report.append(str(train_data.shape)+"\n")

report.append("\nTesting data shape:\n")
report.append(str(test_data.shape)+"\n")

report.append("\nModel structure:\n")
report.append("Input layer based on the number of features\n")
report.append("Hidden layer with 128 neurons and relu activation\n")
report.append("Hidden layer with 64 neurons and relu activation\n")
report.append("Hidden layer with 32 neurons and relu activation\n")
report.append("Output layer with 1 neuron for price prediction\n")

report.append("\nEvaluation results:\n")
report.append("MAE: "+str(mae)+"\n")
report.append("MSE: "+str(mse)+"\n")
report.append("RMSE: "+str(rmse)+"\n")
report.append("R2: "+str(r2)+"\n")

report.append("\nBusiness interpretation:\n")
report.append("This model can be used to estimate the price of a diamond based on its features.\n")
report.append("MAE and RMSE show the average size of the prediction error in dollars.\n")
report.append("R2 shows how much of the price variation is explained by the model.\n")
report.append("The scatter plot with the ideal line helps show how close the predicted prices are to the real prices.\n")
report.append("The comparison plot shows actual and predicted prices for the first 200 test samples using both lines and markers.\n")

report.append("\nSaved output files:\n")
report.append("regression_metrics.csv\n")
report.append("regression_predictions.csv\n")
report.append("training_history.csv\n")
report.append("regression_model_structure.txt\n")
report.append("training_validation_loss.png\n")
report.append("training_validation_mae.png\n")
report.append("actual_vs_predicted_price_scatter.png\n")
report.append("actual_vs_predicted_price_comparison.png\n")
report.append("prediction_error_distribution.png\n")
report.append("regression_metrics_plot.png\n")
report.append("price_ann_model.keras\n")
report.append("regression_model_report.txt\n")

report.append("\nStep 7 is complete.\n")

with open(
    os.path.join(output_folder,"regression_model_report.txt"),
    "w",
    encoding="utf-8") as file:
    file.writelines(report)


# 15. Final message

print("\nStep 7 - Build the Regression Model is complete.")
print("All output files were saved inside this folder:")
print(output_folder)