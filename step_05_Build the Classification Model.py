# Step 5 - Build the Classification Model
# In this step, I build an Artificial Neural Network (ANN)
# to classify diamond clarity.
# I train the model using the prepared dataset from Step 4.
# Then I evaluate the model using accuracy, precision, recall, and F1-score.
# I also save useful plots to show the model results.

import os

# These lines reduce TensorFlow warning messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import tensorflow as tf

tf.get_logger().setLevel("ERROR")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input


# 1. Folder paths


input_folder = "step_04_Data Preprocessing and Feature Engineering"
output_folder = "step_05_Build the Classification Model"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


train_file = os.path.join(input_folder, "classification_train.csv")
test_file = os.path.join(input_folder, "classification_test.csv")
mapping_file = os.path.join(input_folder, "clarity_label_mapping.csv")



# 2. Load the prepared classification data


train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)
clarity_mapping = pd.read_csv(mapping_file)

print("Training and testing data loaded successfully.")
print("Training data shape:", train_data.shape)
print("Testing data shape:", test_data.shape)



# 3. Split features and target


target_column = "clarity_encoded"

X_train = train_data.drop(columns=[target_column]) #creating the training inputs.
y_train = train_data[target_column].astype(int)  #creating the training target.

X_test = test_data.drop(columns=[target_column])
y_test = test_data[target_column].astype(int)


number_of_features = X_train.shape[1]
number_of_classes = clarity_mapping.shape[0]

print("Number of features:", number_of_features)
print("Number of clarity classes:", number_of_classes)



# 4. Build the ANN classification model


# This is a simple ANN model.
# The last layer uses softmax because this is a multi-class classification problem.

np.random.seed(42)
tf.random.set_seed(42)

model = Sequential()

model.add(Input(shape=(number_of_features,)))
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(number_of_classes, activation="softmax"))


model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])



# 5. Train the model
#validation_split=0.2: taking 20% of the training data and use it as validation data during training.
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)



# 6. Make predictions


y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1) #choosesing the class with the highest probability.



# 7. Evaluate the model


accuracy = accuracy_score(y_test, y_pred)


#"weighted":::; calculating precision/recall/f1 for all clarity classes, 
#then combining them while giving more weight to bigger classes
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)

recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)

f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print("\nModel Evaluation Results")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)



# 8. Save evaluation results


metrics_table = pd.DataFrame({"Metric": ["Accuracy", "Precision", "Recall", "F1-score"], "Score": [accuracy, precision, recall, f1]})

metrics_table.to_csv(os.path.join(output_folder, "classification_metrics.csv"), index=False)


class_names = clarity_mapping["clarity"].tolist()

class_report = classification_report(y_test, y_pred, labels=list(range(number_of_classes)), target_names=class_names,
    zero_division=0)

with open(os.path.join(output_folder, "classification_report.txt"), "w", encoding="utf-8") as file:
    file.write(class_report)


conf_matrix = confusion_matrix(y_test, y_pred, labels=list(range(number_of_classes)))

confusion_matrix_table = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

confusion_matrix_table.to_csv(os.path.join(output_folder, "confusion_matrix.csv"))



# 9. Save predictions


encoded_to_clarity = dict(zip(clarity_mapping["clarity_encoded"], clarity_mapping["clarity"]))

actual_clarity = y_test.map(encoded_to_clarity)
predicted_clarity = pd.Series(y_pred).map(encoded_to_clarity)

predictions = pd.DataFrame({"actual_clarity_encoded": y_test.values, "predicted_clarity_encoded": y_pred,
    "actual_clarity": actual_clarity.values,
    "predicted_clarity": predicted_clarity.values})

predictions.to_csv(os.path.join(output_folder, "classification_predictions.csv"),
    index=False)



# 10. Save training history


history_table = pd.DataFrame(history.history)

history_table.to_csv(os.path.join(output_folder, "training_history.csv"), index=False)



# 11. Save useful plots


# Plot training accuracy and validation accuracy

plt.figure(figsize=(8, 5))
plt.plot(history_table["accuracy"], label="Training accuracy")
plt.plot(history_table["val_accuracy"], label="Validation accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "training_validation_accuracy.png"))
plt.close()


# Plot training loss and validation loss

plt.figure(figsize=(8, 5))
plt.plot(history_table["loss"], label="Training loss")
plt.plot(history_table["val_loss"], label="Validation loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "training_validation_loss.png"))
plt.close()


# Plot classification metrics

plt.figure(figsize=(8, 5))
sns.barplot(x="Metric", y="Score", data=metrics_table)
plt.title("Classification Model Metrics")
plt.xlabel("Metric")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "classification_metrics_plot.png"))
plt.close()


# Plot confusion matrix

plt.figure(figsize=(9, 7))
sns.heatmap(confusion_matrix_table, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Clarity")
plt.ylabel("Actual Clarity")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "confusion_matrix_plot.png"))
plt.close()


# Plot actual clarity count and predicted clarity count

actual_counts = predictions["actual_clarity"].value_counts().sort_index()
predicted_counts = predictions["predicted_clarity"].value_counts().sort_index()

actual_predicted_counts = pd.DataFrame({"Actual": actual_counts, "Predicted": predicted_counts})

actual_predicted_counts = actual_predicted_counts.fillna(0)

actual_predicted_counts.to_csv(
    os.path.join(output_folder, "actual_vs_predicted_clarity_counts.csv"))

actual_predicted_counts.plot(kind="bar", figsize=(9, 5))
plt.title("Actual vs Predicted Clarity Distribution")
plt.xlabel("Clarity")
plt.ylabel("Number of Diamonds")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "actual_vs_predicted_clarity_distribution.png"))
plt.close()



# 12. Save the trained model


model.save(os.path.join(output_folder, "clarity_ann_model.keras"))



# 13. Save a simple report in my own words


report = []

report.append("Step 5 - Build the Classification Model\n")
report.append("\nIn this step, I built an Artificial Neural Network model to classify diamond clarity.\n")
report.append("I used the prepared classification training and testing files from Step 4.\n")
report.append("The target column for this model is clarity_encoded.\n")

report.append("\nTraining data shape:\n")
report.append(str(train_data.shape) + "\n")

report.append("\nTesting data shape:\n")
report.append(str(test_data.shape) + "\n")

report.append("\nModel structure:\n")
report.append("Input layer based on the number of features\n")
report.append("Hidden layer with 128 neurons and relu activation\n")
report.append("Hidden layer with 64 neurons and relu activation\n")
report.append("Hidden layer with 32 neurons and relu activation\n")
report.append("Output layer with softmax activation\n")

report.append("\nEvaluation results:\n")
report.append("Accuracy: " + str(accuracy) + "\n")
report.append("Precision: " + str(precision) + "\n")
report.append("Recall: " + str(recall) + "\n")
report.append("F1-score: " + str(f1) + "\n")

report.append("\nBusiness interpretation:\n")
report.append("This model can be used to estimate the clarity category of a diamond based on its features.\n")
report.append("This can help organize diamond information and support pricing or product analysis.\n")
report.append("The confusion matrix is useful because it shows which clarity groups the model predicts well and which groups it mixes up.\n")
report.append("The training and validation plots help me check whether the model is learning properly or overfitting.\n")

report.append("\nSaved output files:\n")
report.append("classification_metrics.csv\n")
report.append("classification_report.txt\n")
report.append("confusion_matrix.csv\n")
report.append("classification_predictions.csv\n")
report.append("training_history.csv\n")
report.append("actual_vs_predicted_clarity_counts.csv\n")
report.append("training_validation_accuracy.png\n")
report.append("training_validation_loss.png\n")
report.append("classification_metrics_plot.png\n")
report.append("confusion_matrix_plot.png\n")
report.append("actual_vs_predicted_clarity_distribution.png\n")
report.append("clarity_ann_model.keras\n")
report.append("classification_model_report.txt\n")

report.append("\nStep 5 is complete.\n")

with open(
    os.path.join(output_folder, "classification_model_report.txt"),
    "w",
    encoding="utf-8") as file:
    file.writelines(report)



# 14. Final message


print("\nStep 5 - Build the Classification Model is complete.")
print("All output files were saved inside this folder:")
print(output_folder)