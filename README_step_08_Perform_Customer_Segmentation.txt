Step 8 - Perform Customer Segmentation

In this step, I used K-Means clustering to group diamonds into different segments.

Code file:
step_08_Perform Customer Segmentation.py

Input folder:
step_04_Data Preprocessing and Feature Engineering

Input file:
feature_engineered_dataset.csv

Libraries used:
pandas
numpy
os
matplotlib
scikit-learn

Output folder:
step_08_Perform Customer Segmentation

What this step does:
I selected numerical features such as carat, depth, table, price, x, y, z, and volume. I scaled the features, used the elbow method to check the number of clusters, and then applied K-Means clustering with 4 clusters.

Main outputs:
- diamonds_with_clusters.csv: dataset with a new cluster column
- cluster_summary.csv: average values for each cluster
- elbow_method_values.csv: inertia values for different cluster numbers
- pca_cluster_data.csv: PCA data used for the 2D cluster plot
- elbow_method_plot.png: elbow method plot
- cluster_visualization_pca.png: PCA cluster visualization
- cluster_size_plot.png: number of diamonds in each cluster
- average_price_by_cluster.png: average price for each cluster
- average_carat_by_cluster.png: average carat for each cluster
- customer_segmentation_report.txt: summary of clustering results

Why this step matters:
This step helps me group diamonds into different segments based on size, price, and shape features. Since clustering is unsupervised, I do not use accuracy. I use the elbow method and cluster summaries to understand the result.