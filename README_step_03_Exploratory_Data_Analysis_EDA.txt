README_step_03_Exploratory_Data_Analysis_EDA.txt

Step 3 - Exploratory Data Analysis (EDA)

In this step, I explored the Diamonds dataset before preprocessing and modeling.

Code file:
step_03_Exploratory Data Analysis (EDA).py

Input file:
diamonds.csv

Input location:
diamonds.csv should be in the same folder as the code.

Libraries used:
pandas
matplotlib
seaborn
os

Output folder:
step_03_Exploratory Data Analysis (EDA)

Main outputs:
- eda_report.txt: summary of the EDA results
- missing_values.csv: missing values in each column
- numeric_summary.csv: descriptive statistics
- correlation_matrix.csv: correlation values
- correlation_matrix.png: heatmap of numerical correlations
- distribution plots: carat, depth, table, price, x, y, and z
- carat_vs_price.png: relationship between carat and price
- price_by_cut.png: price comparison by cut
- price_by_color.png: price comparison by color
- price_by_clarity.png: price comparison by clarity
- average_price_by_cut.csv: average price for each cut
- average_price_by_color.csv: average price for each color
- average_price_by_clarity.csv: average price for each clarity
- rows_with_zero_dimensions.csv: rows where x, y, or z is zero

Main findings:
The dataset has no missing values.
There are duplicate rows.
Some rows have zero values for diamond dimensions.
Carat has a strong relationship with price.
Price is not evenly distributed, with many lower-priced diamonds and fewer expensive diamonds.

Why this step matters:
This step helped me understand the dataset, find possible data issues, and prepare for preprocessing and machine learning.