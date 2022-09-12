
'''
	Taking in feature vectors from model and visualising using UMAP - PegNet (ResNet50), day time images
	Running kmeans clustering

	Peggy Bevan Aug 2022
'''

print('loading packages')
import numpy as np
import pickle
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn' #stop pd warning about chain indexing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
#!pip install umap-learn
import umap
import json
import hdbscan
import sklearn.cluster as cluster
from collections import defaultdict
from sklearn import metrics, datasets
from sklearn.metrics import pairwise_distances, adjusted_rand_score, adjusted_mutual_info_score

#%matplotlib auto #display plots


#read in numpy arrays - output from predict.py
print('reading in feature vectors')
features_PN = np.load('output/PegNet_fvect_norm.npy')
imgs_PN = np.load('output/PegNet_imgvect_norm.npy')

#add information about camera trap location and species
meta = pd.read_csv('data/nepal_cropsmeta_PB.csv')

#meta has humans and vehicles in - remove
anthro = ['human', 'vehicle']
meta = meta[-meta['species'].isin(anthro)]
#only training
meta = meta[meta['SetID']=='train']

#create numpy array for important variables
ct_site = np.array(meta.ct_site)
species = np.array(meta.species)
mgmt = np.array(meta.conservancy_name) 
time_hour = np.array(meta.time_hour)
#it would be good to add land cover type
#higher functional classification 
	# canids, felids, small herbivore, large herbivore, domestic animal
print('removing nocturnal images')
dayfeatures = []
dayhour = []
dayspecies = []
daymgmt = []
daysite = []
for i, v in enumerate(features_PN):
	if time_hour[i] >= 6 and time_hour[i] <= 18:
		dayfeatures.append(v)
		dayhour.append(time_hour[i])
		dayspecies.append(species[i])
		daymgmt.append(mgmt[i])
		daysite.append(ct_site[i])

dayfeatures  = np.vstack(dayfeatures) #turn back into array
dayhour = np.array(dayhour)
dayspecies = np.array(dayspecies)
daymgmt = np.array(daymgmt)
daysite = np.array(daysite, dtype = 'object')

vecsbysite = {}
for site in sorted(set(daysite)):
	vecsbysite[site] = dayfeatures[daysite==site]

specsbysite = {}
for site in sorted(set(daysite)):
	specsbysite[site] = dayspecies[daysite==site]


#kcomp is a dataframe with numimgs, nspecies, and a column called 'OK_dim' for each dimension
#dimensions is a list
#vecs & specs by site is a dictionary with list of vectors and labels in the same order
def findk(kcomp, dimensions, vecsbysite, specsbysite, kmax=15):
	for index,row in kcomp.iterrows():
		site = row['site']
		print(site)
		x = vecsbysite[site]
		y = specsbysite[site]
		kcomp['numimgs'][index] = len(x)
		kcomp['nspecies'][index] = len(set(y))
		#finding optimal value of k using the silhouette coefficient
		#if number of samples is less than kmax, change optimK
		for dim in dimensions:
			sil = []
			print(f'Finding optimal k for site {site} with {dim} dimensions')
			if len(x) > 5: #the 5 is arbitrary
				embedding = umap.UMAP(init = 'random', n_components=dim).fit_transform(x)
				if len(x) <= kmax:
					kmax = len(x)-1
					for k in range(2, kmax+1):
						print(f'k = {k}')
						kmeans = cluster.KMeans(n_clusters = k).fit(embedding)
						labels = kmeans.labels_ #underscore is needed
						sil.append(metrics.silhouette_score(embedding, labels, metric = 'euclidean'))
					#find index of max silhouette and add 2 (0 = 2)	
					optimk = sil.index(max(sil))+2
				else:
					for k in range(2, kmax+1):
						print(f'k = {k}')
						kmeans = cluster.KMeans(n_clusters = k).fit(embedding)
						labels = kmeans.labels_ #underscore is needed
						sil.append(metrics.silhouette_score(embedding, labels, metric = 'euclidean'))
						#find index of max silhouette and add 2 (0 = 2)
					optimk = sil.index(max(sil))+2
			else:
				optimk = 'NA'
			column = ('OK_' + str(dim))
			kcomp[column][index] = optimk
			print(f'Optimal cluster number = {optimk}')

k_PN_day = pd.DataFrame(sorted(set(daysite)), columns = ['site'])
k_PN_day['numimgs'] = pd.NaT
k_PN_day['nspecies'] = pd.NaT
k_PN_day['OK_2048'] = pd.NaT

dimensions = [2048]
test = findk(kcomp = k_PN_day, dimensions = dimensions, vecsbysite = vecsbysite, specsbysite = specsbysite)

k_PN_day.to_csv('output/kmeans_PN_day.csv', index=False)
