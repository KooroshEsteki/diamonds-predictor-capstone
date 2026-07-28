# Step 8 - Perform Customer Segmentation
# In this step, I apply clustering to segment diamonds based on selected features.
# I use K-Means clustering, analyze the clusters, and visualize the results.

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# 1. Folder paths

input_folder = "step_04_Data Preprocessing and Feature Engineering"
output_folder = "step_08_Perform Customer Segmentation"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


data_file = os.path.join(input_folder, "feature_engineered_dataset.csv")


# 2. Load the dataset

df = pd.read_csv(data_file)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# 3. Select features for clustering

# I selected numerical features that describe diamond size, shape, and price.
# These features are useful for grouping diamonds into different market segments.


#K-Means will group diamonds based on similarity in these columns.
selected_features = ["carat", "depth", "table", "price", "x", "y", "z", "volume"]

## creating a new dataset containing only the selected clustering features.
clustering_data = df[selected_features].copy()  

print("\nSelected features for clustering:")
print(selected_features)


# 4. Scale the selected features

scaler = StandardScaler()
clustering_scaled = scaler.fit_transform(clustering_data)


# 5. Use elbow method to check a reasonable number of clusters

inertia_values = []

k_values = range(1, 11)

# n_init=10:::: K-Means tries 10 different starting points and keeps the best result.
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

    kmeans.fit(clustering_scaled)
    inertia_values.append(kmeans.inertia_)


elbow_table = pd.DataFrame({"number_of_clusters": list(k_values), "inertia": inertia_values})

elbow_table.to_csv(os.path.join(output_folder, "elbow_method_values.csv"), index=False)


plt.figure(figsize=(8, 5))
plt.plot(list(k_values), inertia_values, marker="o")
plt.title("Elbow Method for K-Means Clustering")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "elbow_method_plot.png"))
plt.close()


# 6. Build the final clustering model

# I use 4 clusters as a simple and reasonable choice for segmentation.

number_of_clusters = 4

kmeans_final = KMeans(n_clusters=number_of_clusters, random_state=42, n_init=10)

cluster_labels = kmeans_final.fit_predict(clustering_scaled)

df["cluster"] = cluster_labels


# 7. Save the dataset with cluster labels

df.to_csv(os.path.join(output_folder, "diamonds_with_clusters.csv"), index=False)


# 8. Analyze the clusters

cluster_summary = df.groupby("cluster")[selected_features].mean()
cluster_summary["number_of_diamonds"] = df["cluster"].value_counts().sort_index()

cluster_summary.to_csv(os.path.join(output_folder, "cluster_summary.csv"))

print("\nCluster summary:")
print(cluster_summary)


# 9. Use PCA only for 2D visualization

pca = PCA(n_components=2)
pca_result = pca.fit_transform(clustering_scaled)

pca_df = pd.DataFrame({"PCA1": pca_result[:, 0],"PCA2": pca_result[:, 1],"cluster": cluster_labels})

pca_df.to_csv(os.path.join(output_folder, "pca_cluster_data.csv"),index=False)


# 10. Visualize clustering results

plt.figure(figsize=(8, 6))

for cluster_number in range(number_of_clusters):
    cluster_points = pca_df[pca_df["cluster"] == cluster_number]

    plt.scatter(cluster_points["PCA1"],cluster_points["PCA2"],label="Cluster " + str(cluster_number),
        alpha=0.6,s=20)

plt.title("Customer Segmentation Using K-Means Clustering")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "cluster_visualization_pca.png"))
plt.close()


# Cluster size plot

cluster_counts = df["cluster"].value_counts().sort_index()

plt.figure(figsize=(7, 5))
plt.bar(cluster_counts.index,cluster_counts.values)
plt.title("Number of Diamonds in Each Cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of Diamonds")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "cluster_size_plot.png"))
plt.close()


# Average price by cluster

avg_price_by_cluster = df.groupby("cluster")["price"].mean()

plt.figure(figsize=(7, 5))
plt.bar(avg_price_by_cluster.index,avg_price_by_cluster.values)
plt.title("Average Price by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Average Price")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "average_price_by_cluster.png"))
plt.close()


# Average carat by cluster

avg_carat_by_cluster = df.groupby("cluster")["carat"].mean()

plt.figure(figsize=(7, 5))
plt.bar(avg_carat_by_cluster.index,avg_carat_by_cluster.values)
plt.title("Average Carat by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Average Carat")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "average_carat_by_cluster.png"))
plt.close()


# 11. Save a simple interpretation report

report = []

report.append("Step 8 - Perform Customer Segmentation\n")
report.append("\nIn this step, I used K-Means clustering to segment diamonds based on selected numerical features.\n")
report.append("The selected features were carat, depth, table, price, x, y, z, and volume.\n")
report.append("I scaled the features first because clustering is affected by the size of numerical values.\n")

report.append("\nNumber of clusters used:\n")
report.append(str(number_of_clusters) + "\n")

report.append("\nCluster summary:\n")
report.append(str(cluster_summary) + "\n")

report.append("\nMy interpretation:\n")

for cluster_number in range(number_of_clusters):
    cluster_info = cluster_summary.loc[cluster_number]

    report.append("\nCluster " + str(cluster_number) + ":\n")
    report.append("Number of diamonds: " + str(int(cluster_info["number_of_diamonds"])) + "\n")
    report.append("Average price: $" + str(round(cluster_info["price"], 2)) + "\n")
    report.append("Average carat: " + str(round(cluster_info["carat"], 2)) + "\n")
    report.append("Average volume: " + str(round(cluster_info["volume"], 2)) + "\n")

    if cluster_info["price"] == cluster_summary["price"].max():
        report.append("This cluster has the highest average price, so it can represent a more expensive diamond segment.\n")
    elif cluster_info["price"] == cluster_summary["price"].min():
        report.append("This cluster has the lowest average price, so it can represent a lower-price diamond segment.\n")
    else:
        report.append("This cluster represents a middle price segment compared with the other groups.\n")

report.append("\nBusiness interpretation:\n")
report.append("The clusters can help separate diamonds into different groups based on size, price, and shape-related features.\n")
report.append("This can be useful for understanding different product segments, such as lower-price diamonds, mid-range diamonds, and higher-price diamonds.\n")
report.append("The cluster plots help visualize how the diamonds are grouped after reducing the data to two PCA components.\n")

report.append("\nSaved output files:\n")
report.append("diamonds_with_clusters.csv\n")
report.append("cluster_summary.csv\n")
report.append("elbow_method_values.csv\n")
report.append("pca_cluster_data.csv\n")
report.append("elbow_method_plot.png\n")
report.append("cluster_visualization_pca.png\n")
report.append("cluster_size_plot.png\n")
report.append("average_price_by_cluster.png\n")
report.append("average_carat_by_cluster.png\n")
report.append("customer_segmentation_report.txt\n")

report.append("\nStep 8 is complete.\n")

with open(os.path.join(output_folder, "customer_segmentation_report.txt"),
    "w",encoding="utf-8") as file: file.writelines(report)


# 12. Final message

print("\nStep 8 - Perform Customer Segmentation is complete.")
print("All output files were saved inside this folder:")
print(output_folder)