# Part 2: Cluster Analysis

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import matplotlib

# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'wholesale_customers.csv'.
def read_csv_2(data_file):
	df = pd.read_csv(data_file)
	df = df.drop('Channel', axis = 1)
	df = df.drop('Region', axis = 1)
	return df

# Return a pandas dataframe with summary statistics of the data.
# Namely, 'mean', 'std' (standard deviation), 'min', and 'max' for each attribute.
# These strings index the new dataframe columns. 
# Each row should correspond to an attribute in the original data and be indexed with the attribute name.
def summary_statistics(df):
	return df.describe().transpose().round(0).astype(int)[['mean', 'std', 'min', 'max']]

# Given a dataframe df with numeric values, return a dataframe (new copy)
# where each attribute value is subtracted by the mean and then divided by the
# standard deviation for that attribute.
def standardize(df):
	df_std = df.copy(deep = True)
	mean = df_std.mean()
	std_dev = df_std.std()
	df_std = (df_std - mean)/std_dev
	
	return df_std

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans.
# y should contain values in the set {0,1,...,k-1}.
# To see the impact of the random initialization,
# using only one set of initial centroids in the kmeans run.
def kmeans(df, k):
	k_means = KMeans(n_clusters = k, n_init=1)
	cluster_labels = k_means.fit_predict(df)
	return pd.Series(cluster_labels)

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans++.
# y should contain values from the set {0,1,...,k-1}.
def kmeans_plus(df, k):
	kmeans = KMeans(n_clusters = k, init='k-means++')
	cluster_labels = kmeans.fit_predict(df)
	return pd.Series(cluster_labels)

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using agglomerative hierarchical clustering.
# y should contain values from the set {0,1,...,k-1}.
def agglomerative(df, k):
	hierarchical_cluster = AgglomerativeClustering(n_clusters=k, metric='euclidean', linkage='ward')
	cluster_labels = hierarchical_cluster.fit_predict(df)
	return pd.Series(cluster_labels)

# Given a data set X and an assignment to clusters y
# return the Silhouette score of this set of clusters.
def clustering_score(X,y):
	return silhouette_score(X, y)

# Perform the cluster evaluation described in the coursework description.
# Given the dataframe df with the data to be clustered,
# return a pandas dataframe with an entry for each clustering algorithm execution.
# Each entry should contain the: 
# 'Algorithm' name: either 'Kmeans' or 'Agglomerative', 
# 'data' type: either 'Original' or 'Standardized',
# 'k': the number of clusters produced,
# 'Silhouette Score': for evaluating the resulting set of clusters.
def cluster_evaluation(df):
	algorithm = ['Kmeans', 'Agglomerative']
	data_type = ['Original', 'Standardized']
	no_of_clusters = [2,3,5]
	scores = []
	df_std = standardize(df)
	for algo in algorithm:
		for data in data_type:
			if data == 'Standard':
				X = df_std
			else:
				X = df
			for k in no_of_clusters:
				if algo == 'Kmeans':
					for _ in range(10):
						scores.append([algo, data, k, clustering_score(X, kmeans(df, k))])
				else:
					scores.append([algo, data, k, clustering_score(X, agglomerative(df, k))])
	
	return pd.DataFrame(scores, columns=['Algorithm', 'data', 'k', 'Sillhouette Score'])



# Given the performance evaluation dataframe produced by the cluster_evaluation function,
# return the best computed Silhouette score.
def best_clustering_score(rdf):
	return rdf['Sillhouette Score'].max()

# Run the Kmeans algorithm with k=3 by using the standardized data set.
# Generate a scatter plot for each pair of attributes.
# Data points in different clusters should appear with different colors.
def scatter_plots(df):
	df_std = standardize(df)
	cluster_labels = kmeans(df_std, 3)
	i = 0
	j = 0
	plt.figure(figsize=(8, 6))
	for i in range(len(df_std.columns)):
		for j in range(i+1, len(df_std.columns)):
			for cluster in range(3):
				plt.scatter(df_std[cluster_labels == cluster].iloc[:, i], df_std[cluster_labels == cluster].iloc[:, j], label=f'Cluster {cluster}') 
			plt.xlabel(df_std.columns[i])
			plt.ylabel(df_std.columns[j])
			plt.title(f'Scatter Plot of {df_std.columns[i]} vs {df_std.columns[j]}')
			plt.legend()
			plt.savefig(f'scatterplot_{df_std.columns[i]}_vs_{df_std.columns[j]}.pdf', format='pdf', bbox_inches='tight')
			plt.close()