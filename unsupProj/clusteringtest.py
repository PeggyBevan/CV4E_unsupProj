##clusteringtest.py

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
print('loading packages')
import numpy as np
import pickle
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn' #stop pd warning about chain indexing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
#!conda install -c conda-forge umap-learn
import umap
from umap import plot
import json
import hdbscan
import sklearn.cluster as cluster
from collections import defaultdict
from sklearn import metrics, datasets
from sklearn.metrics import pairwise_distances, adjusted_rand_score, adjusted_mutual_info_score
import colorcet as cc
from umap import plot

#%matplotlib auto #display plots


#read in numpy arrays - output from predict.py
print('reading in feature vectors')
features_PN = np.load('output/PegNet_fvect_norm.npy')
imgs_PN = np.load('output/PegNet_imgvect_norm.npy')

#add information about camera trap location and species
meta = pd.read_csv('data/nepal_cropsmeta_PB.csv')

#meta has humans and vehicles in - remove
anthro = ['human', 'vehicle']
domes = ['buffalo', 'cow', 'dog', 'domestic_cat', 'domestic_chicken', 'domestic elephant', 'goat', 'sheep']
meta = meta[-meta['species'].isin(anthro)]
#only training
meta = meta[meta['SetID']=='train']
meta_wild = meta[-meta['species'].isin(domes)]

#create numpy array for important variables
ct_site = np.array(meta.ct_site)
species = np.array(meta.species)
mgmt = np.array(meta.conservancy_name) 
time_hour = np.array(meta.time_hour)
#it would be good to add land cover type
#higher functional classification 
	# canids, felids, small herbivore, large herbivore, domestic animal

#####----Visualising entire dataset - PegNet-----######
#create umap object

###Assign a number of clusters to a site, and work out how many clusters each camera trap site falls into.
print('first dimension reduction')
embedding = umap.UMAP(init = 'random', random_state=42, n_components=2).fit_transform(features_PN)
#kmeans = cluster.KMeans(n_clusters = 20, n_init = 10)
#print('attempting kmeans')
#labels = kmeans.fit_predict(embedding)
#labels = kmeans.labels_ 
#print(labels)

print('attempting hdbscan')
labels = hdbscan.HDBSCAN(
    min_samples=80,
    min_cluster_size=80,
    cluster_selection_method = 'leaf'
).fit_predict(embedding)
labels.max()
#note - tried different metrics (manhattan, hamming) they all made no difference.

clustered = (labels >= 0)

plt.scatter(embedding[~clustered, 0],
            embedding[~clustered, 1],
            color=(0.5, 0.5, 0.5),
            s=0.1,
            alpha=0.5)
plt.scatter(embedding[clustered, 0],
            embedding[clustered, 1],
            c=labels[clustered],
            s=0.1,
            cmap='Spectral')
#plt.show()
print(labels.max())
#how do the camera trap sites match up with cluster labels?
plt.scatter(embedding[ :, 0],
            embedding[ :, 1],
            color=species.astype(int),
            s=0.1
            )
sns.scatterplot(x=embedding[ :, 0], y=embedding[ :, 1], hue=species)

#at the moment, the clusters are simply not grouping by species, so this is not very helpful. I need a fine-tuned model that can perform a bit better at recognising animals. 

# Create a dictionary to store counts of distinct labels for each site
site_label_counts = {}

# Iterate over unique sites in ct_site
for site in np.unique(ct_site):
    # Get the indices corresponding to the current site
    indices = np.where(ct_site == site)[0]
    # Extract labels for the current site
    site_labels = labels[indices]
    site_species = species[indices]
    site_labels = site_labels[site_labels != -1] #remove imgs labelled as 'noise'
    # Count the number of distinct labels for the current site
    distinct_label_count = len(np.unique(site_labels))
    distinct_species_count = len(np.unique(site_species))
    # Store the count in the dictionary
    site_label_counts[site] = {'distinct_label_count': distinct_label_count,
                               'distinct_species_count': distinct_species_count}

# If you want the results as an array, you can convert the dictionary to a numpy array
result_array = np.array(list(site_label_counts.items()), dtype=int)

distinct_label_counts = [counts['distinct_label_count'] for counts in site_label_counts.values()]
distinct_species_counts = [counts['distinct_species_count'] for counts in site_label_counts.values()]

# Fit a linear regression line
regression_coefficients = np.polyfit(distinct_label_counts, distinct_species_counts, 1)
regression_line = np.polyval(regression_coefficients, distinct_label_counts)

# Calculate the correlation coefficient
correlation_coefficient = np.corrcoef(distinct_label_counts, distinct_species_counts)[0, 1]

# Plot the relationship
plt.scatter(distinct_label_counts, distinct_species_counts)
plt.plot(distinct_label_counts, regression_line, color='red', label='Regression Line')
plt.title('Relationship between Distinct Label Count and Distinct Species Count')
plt.xlabel('Distinct Clusters per CT site')
plt.ylabel('Species Count per CT site')
plt.legend(['Data Points (r={:.2f})'.format(correlation_coefficient), 'Regression Line'])

plt.show()
plt.savefig('Output/figs/PegNet/umap_hdbscan_clusterspecies_correlation.png')
#there is a 0.4 correlation!

labels = hdbscan.HDBSCAN(
    min_samples=10,
    min_cluster_size=15,
    cluster_selection_method = 'leaf'
).fit_predict(embedding)
labels.max()

# Create a dictionary to store counts of distinct labels for each site
site_label_counts2 = {}

# Iterate over unique sites in ct_site
for site in np.unique(ct_site):
    # Get the indices corresponding to the current site
    indices = np.where(ct_site == site)[0]
    # Extract labels for the current site
    site_labels = labels[indices]
    site_species = species[indices]
    # Count the number of distinct labels for the current site
    distinct_label_count = len(np.unique(site_labels))
    distinct_species_count = len(np.unique(site_species))
    # Store the count in the dictionary
    site_label_counts2[site] = {'distinct_label_count': distinct_label_count,
                               'distinct_species_count': distinct_species_count}

# If you want the results as an array, you can convert the dictionary to a numpy array
result_array = np.array(list(site_label_counts.items()), dtype=int)

distinct_label_counts = [counts['distinct_label_count'] for counts in site_label_counts2.values()]
distinct_species_counts = [counts['distinct_species_count'] for counts in site_label_counts2.values()]

# Fit a linear regression line
regression_coefficients = np.polyfit(distinct_label_counts, distinct_species_counts, 1)
regression_line = np.polyval(regression_coefficients, distinct_label_counts)

# Calculate the correlation coefficient
correlation_coefficient = np.corrcoef(distinct_label_counts, distinct_species_counts)[0, 1]

# Plot the relationship
plt.scatter(distinct_label_counts, distinct_species_counts)
plt.plot(distinct_label_counts, regression_line, color='red', label='Regression Line')

plt.title('Relationship between Distinct Label Count and Distinct Species Count')
plt.xlabel('Distinct Clusters per CT site')
plt.ylabel('Species Count per CT site')
plt.legend(['Data Points (r={:.2f})'.format(correlation_coefficient), 'Regression Line'])

plt.savefig('Output/figs/PegNet/umap_hdbscan_clusterspecies_10_15.png')