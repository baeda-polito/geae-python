# Part 1: Importing Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score

# Part 2: Loading Data
# Replace 'pydata.csv' and 'df_info.csv' with your actual data file paths
pydata = pd.read_csv('data/pydata.csv', parse_dates=['date_time'])
df_info = pd.read_csv('data/df_info.csv')

# Part 3: Introducing Missing Values
# Randomly select 200 indices to set as NaN in 'energy_h' column
np.random.seed(0)  # For reproducibility
missing_indices = np.random.choice(pydata.index, size=200, replace=False)
pydata.loc[missing_indices, 'energy_h'] = np.nan

# Check the number of missing values
print("Number of missing values in 'energy_h':", pydata['energy_h'].isna().sum())

# Part 4: Visualizing Original Time Series
# Create 'date' and 'time' columns if not already present
pydata['date'] = pydata['date_time'].dt.date
pydata['time'] = pydata['date_time'].dt.time

# Pivot the data for heatmap
pivot_original = pydata.pivot_table(index='date', columns='time', values='energy_h')

# Plotting the carpet plot
plt.figure(figsize=(15, 10))
sns.heatmap(pivot_original, cmap='Spectral_r', cbar_kws={'label': 'Electrical energy [kWh]'},
            xticklabels=48, yticklabels=30)
plt.title('Carpet Plot of Energy Consumption (Original Data)')
plt.xlabel('Time of Day')
plt.ylabel('Date')
plt.show()

# Part 5: Imputing Missing Values
pydata['energy_h'] = pydata['energy_h'].interpolate(method='linear')

# Verify that there are no missing values
print("Number of missing values after interpolation:", pydata['energy_h'].isna().sum())

# Part 6: Visualizing Cleaned Time Series
# Pivot the data for heatmap
pivot_cleaned = pydata.pivot_table(index='date', columns='time', values='energy_h')

# Plotting the carpet plot
plt.figure(figsize=(15, 10))
sns.heatmap(pivot_cleaned, cmap='Spectral_r', cbar_kws={'label': 'Electrical energy [kWh]'},
            xticklabels=48, yticklabels=30)
plt.title('Carpet Plot of Energy Consumption (Cleaned Data)')
plt.xlabel('Time of Day')
plt.ylabel('Date')
plt.show()

# Part 7: Constructing the M x N Matrix
# Pivoting the data to create a wide-format DataFrame
py_mxn = pydata.pivot(index='date', columns='time', values='energy_h').reset_index()

# Part 8: Normalizing the Data
# Exclude the 'date' column for normalization
data_columns = py_mxn.columns[1:]
py_mxn[data_columns] = py_mxn[data_columns].div(py_mxn[data_columns].max(axis=1), axis=0)

# Part 9: Computing the Distance Matrix
# Calculate the pairwise Euclidean distances
distance_matrix = pdist(py_mxn[data_columns], metric='euclidean')
distance_square_matrix = squareform(distance_matrix)

# Part 10: Hierarchical Clustering
# Perform hierarchical clustering using Ward's method
linked = linkage(distance_matrix, method='ward')

# Plotting the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked, labels=py_mxn['date'].astype(str).values)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Date')
plt.ylabel('Distance')
plt.show()

# Part 11: Determining Optimal Number of Clusters
from sklearn.metrics import silhouette_score

range_n_clusters = list(range(3, 16))
silhouette_avg = []

for n_clusters in range_n_clusters:
    clusterer = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
    cluster_labels = clusterer.fit_predict(py_mxn[data_columns])
    silhouette_avg.append(silhouette_score(py_mxn[data_columns], cluster_labels))

# Plotting silhouette scores
plt.figure(figsize=(8, 5))
plt.plot(range_n_clusters, silhouette_avg, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Average Silhouette Score')
plt.title('Silhouette Analysis For Optimal k')
plt.show()

# Part 12: Assigning Cluster Labels
n_clusters = 4
cluster_model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', metric='euclidean')
py_mxn['cluster'] = cluster_model.fit_predict(py_mxn[data_columns])

# Plotting the dendrogram with cluster cuts
plt.figure(figsize=(10, 7))
dendrogram(linked, labels=py_mxn['date'].astype(str).values, color_threshold=linked[-(n_clusters - 1), 2])
plt.title('Dendrogram with Cluster Cuts')
plt.xlabel('Date')
plt.ylabel('Distance')
plt.axhline(y=linked[-(n_clusters - 1), 2], c='k', ls='--', lw=0.5)
plt.show()

# Part 13: Calculating Cluster Centroids
# Melt the DataFrame to long format for plotting
melted_py_mxn = py_mxn.melt(id_vars=['date', 'cluster'], var_name='time', value_name='norm_energy_h')

# Convert 'time' to datetime objects
melted_py_mxn['time'] = pd.to_datetime(melted_py_mxn['time'], format='%H:%M:%S')

# Calculate statistics for each cluster
centroids = melted_py_mxn.groupby(['cluster', 'time']).agg({
    'norm_energy_h': ['mean', 'std', 'min', 'max']
}).reset_index()
centroids.columns = ['cluster', 'time', 'avg_var', 'sd_var', 'min_var', 'max_var']

# Part 14: Visualizing Clusters and Centroids
# Plot settings
import matplotlib.dates as mdates

# Set the plot style
sns.set(style='whitegrid')


# Create a function to plot in each facet
def facet_plot(data, color):
    cluster = data['cluster'].iloc[0]

    # Plot the daily load profiles in grey, grouped by date
    for date, group in data.groupby('date'):
        plt.plot(group['time'], group['norm_energy_h'], color='grey', alpha=0.3)

    # Get the centroid data for this cluster
    centroid_data = centroids[centroids['cluster'] == cluster]

    # Plot the average line (cluster centroid)
    plt.plot(centroid_data['time'], centroid_data['avg_var'], color='red', linestyle='--', linewidth=2)

    # Plot the standard deviation ribbon
    plt.fill_between(
        centroid_data['time'],
        centroid_data['avg_var'] - centroid_data['sd_var'],
        centroid_data['avg_var'] + centroid_data['sd_var'],
        color='red', alpha=0.2
    )

    # Set the title and labels
    plt.title(f'Cluster {cluster}')
    plt.xlabel('Time')
    plt.ylabel('Normalized Energy')

    # Format x-axis labels for better readability
    plt.xticks(rotation=45)

    # Adjust x-axis formatter to display time properly
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=6))

    # Ensure the layout is tight
    plt.tight_layout()


# Create the FacetGrid
g = sns.FacetGrid(melted_py_mxn, col='cluster', col_wrap=2, height=4, sharey=True)

# Map the plotting function to the FacetGrid
g.map_dataframe(facet_plot)

# Show the plot
plt.show()

# Part 15: Classification Tree
# Prepare data for classification
# Ensure 'day' column has first three letters (e.g., 'Mon', 'Tue')
df_info['day'] = df_info['day'].str[:3]

# For py_mxn
py_mxn['date'] = pd.to_datetime(py_mxn['date'])
# For df_info
df_info['date'] = pd.to_datetime(df_info['date'])

# Merge 'py_mxn' with 'df_info'
classification_data = pd.merge(py_mxn, df_info)


# Features and target variable
X = classification_data[['T_avg', 'T_max', 'T_min', 'day']]
y = classification_data['cluster']

# Convert categorical variable 'day' to numerical using one-hot encoding
X = pd.get_dummies(X, columns=['day'], drop_first=True)

# Initialize the Decision Tree Classifier
clf = DecisionTreeClassifier(min_samples_leaf=10)
clf.fit(X, y)

# Part 16: Plotting the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns, class_names=[str(i) for i in clf.classes_], filled=True)
plt.show()

# Part 17: Model Evaluation
# Add prediction column to 'classification_data'
classification_data['pred'] = clf.predict(X)

# Calculate evaluation metrics
conf_matrix = confusion_matrix(classification_data['cluster'], classification_data['pred'])
accuracy = accuracy_score(classification_data['cluster'], classification_data['pred']) * 100
recall = recall_score(classification_data['cluster'], classification_data['pred'], average='macro') * 100
precision = precision_score(classification_data['cluster'], classification_data['pred'], average='macro') * 100

print("Confusion Matrix:\n", conf_matrix)
print(f"Accuracy: {accuracy:.2f}%")
print(f"Recall: {recall:.2f}%")
print(f"Precision: {precision:.2f}%")
