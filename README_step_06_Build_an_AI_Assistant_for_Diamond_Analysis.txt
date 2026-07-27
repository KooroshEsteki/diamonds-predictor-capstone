Step 6 - Build an AI Assistant for Diamond Analysis

In this step, I built a simple assistant that answers questions about diamonds using the cleaned dataset.

Code file:
step_06_Build an AI Assistant for Diamond Analysis.py

Input folder:
step_04_Data Preprocessing and Feature Engineering

Input file:
feature_engineered_dataset.csv

Libraries used:
pandas
os
re

Output folder:
step_06_Build an AI Assistant for Diamond Analysis

What this step does:
I load the cleaned dataset from Step 4. The assistant checks the question for cut, color, or clarity values, filters the dataset, and gives a simple answer.

Main outputs:
One text file is saved for each question I ask.
Each file includes the question and the answer.

Note:
I used a local dataset-based assistant instead of the OpenAI API because the API requires separate billing/credits.

Why this step matters:
This step shows how I can use the dataset to answer simple diamond-related questions in natural language.