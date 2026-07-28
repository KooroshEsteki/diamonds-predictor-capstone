# Step 6 - Build an AI Assistant for Diamond Analysis
# In this step, I build a simple assistant that answers questions
# about diamonds using the dataset.
# The assistant finds matching diamonds and explains the result.
# Each question and answer is saved in a separate text file.

import pandas as pd
import os
import re


# 1. Folder paths

input_folder = "step_04_Data Preprocessing and Feature Engineering"
output_folder = "step_06_Build an AI Assistant for Diamond Analysis"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

data_file = os.path.join(input_folder, "feature_engineered_dataset.csv")


# 2. Load the dataset

df = pd.read_csv(data_file)

print("Diamond dataset loaded successfully.")
print("Dataset shape:", df.shape)


# 3. Function to check words in the question


# checking whether a specific value exists in the user question.
def contains_value(question, value):
    question = question.lower()
    value = value.lower()
     
    # preparing the word pattern for safe searching
    value_pattern = re.escape(value).replace("\\ ", "\\s+")
    pattern = r"\b" + value_pattern + r"\b"

    return re.search(pattern, question) is not None


# 4. Function to make safe file name

def make_safe_file_name(question):
    file_name = question.strip() #removing extra spaces
    file_name = re.sub(r'[\\/*?:"<>|]', "", file_name)
    file_name = re.sub(r"\s+", " ", file_name)
    file_name = file_name[:120]
    file_name = file_name.strip(" .")

    if file_name == "":
        file_name = "diamond_question"

    return file_name + ".txt"


# 5. Function to avoid overwriting files

def get_unique_file_path(folder, file_name):
    file_path = os.path.join(folder, file_name)

    if not os.path.exists(file_path):
        return file_path

    base_name = os.path.splitext(file_name)[0]
    extension = os.path.splitext(file_name)[1]

    counter = 2

    while True:
        new_file_name = base_name + "_" + str(counter) + extension
        new_file_path = os.path.join(folder, new_file_name)

        if not os.path.exists(new_file_path):
            return new_file_path

        counter = counter + 1


# 6. Find matching diamonds

def find_matching_diamonds(question):
    filtered_df = df.copy()
    filters_used = []

    cut_values = sorted(df["cut"].unique().tolist(), key=len, reverse=True)

    for cut in cut_values:
        if contains_value(question, cut):
            filtered_df = filtered_df[filtered_df["cut"] == cut]
            filters_used.append("cut = " + cut)
            break

    color_values = df["color"].unique().tolist()

    for color in color_values:
        if contains_value(question, color):
            filtered_df = filtered_df[filtered_df["color"] == color]
            filters_used.append("color = " + color)
            break

    clarity_values = sorted(df["clarity"].unique().tolist(), key=len, reverse=True)

    for clarity in clarity_values:
        if contains_value(question, clarity):
            filtered_df = filtered_df[filtered_df["clarity"] == clarity]
            filters_used.append("clarity = " + clarity)
            break

    return filtered_df, filters_used


# 7. Write answer

def explain_results(question, filtered_df, filters_used):
    answer = ""

    answer += "Question: " + question + "\n\n"

    if len(filters_used) > 0:
        answer += "I found diamonds based on these conditions:\n"
        for item in filters_used:
            answer += "- " + item + "\n"
    else:
        answer += "I did not find a specific cut, color, or clarity filter.\n"
        answer += "So I used the full dataset.\n"

    answer += "\nNumber of matching diamonds: " + str(len(filtered_df)) + "\n"

    if len(filtered_df) == 0:
        answer += "\nI could not find diamonds matching this question.\n"
        return answer

    average_price = round(filtered_df["price"].mean(), 2)
    minimum_price = round(filtered_df["price"].min(), 2)
    maximum_price = round(filtered_df["price"].max(), 2)
    average_carat = round(filtered_df["carat"].mean(), 2)

    answer += "\nSummary:\n"
    answer += "Average price: $" + str(average_price) + "\n"
    answer += "Minimum price: $" + str(minimum_price) + "\n"
    answer += "Maximum price: $" + str(maximum_price) + "\n"
    answer += "Average carat: " + str(average_carat) + "\n"

    answer += "\nMy interpretation:\n"
    answer += "Based on the dataset, these diamonds have an average price of about $"
    answer += str(average_price) + ". "

    if average_carat >= 1:
        answer += "The average carat is above 1, so these diamonds are generally larger.\n"
    else:
        answer += "The average carat is below 1, so these diamonds are generally smaller.\n"

    return answer


# 8. Start assistant

print("\nStep 6 - Diamond AI Assistant")
print("You can ask questions about diamonds.")
print("Example: What is the average price of Ideal cut diamonds?")
print("Type exit to stop.\n")


while True:
    # console question uses Python built-in input() function.
    user_question = input("Ask a question about diamonds: ") 

    if user_question.lower() == "exit":
        print("Assistant stopped.")
        break

    matching_diamonds, filters_used = find_matching_diamonds(user_question)

    answer = explain_results(
        user_question,
        matching_diamonds,
        filters_used)

    print("\n" + answer)

    file_name = make_safe_file_name(user_question)
    file_path = get_unique_file_path(output_folder, file_name)

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(answer)

    print("This question and answer were saved in:")
    print(file_path)


print("\nStep 6 - Build an AI Assistant for Diamond Analysis is complete.")
print("All question-answer text files were saved inside this folder:")
print(output_folder)