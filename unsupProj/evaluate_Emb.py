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

features_EB = np.load('output/Emb_fvect.npy')
imgs_EB = np.load('output/Emb_imgvect.npy')

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

#Visualising entire dataset
#create umap object
print('Plotting UMAP embeddings for entire dataset')
fit = umap.UMAP()
u = fit.fit_transform(features_EB) #this line can take a while

##split data by camera trap site
print('Organising data by site')
vecsbysite = {}
for site in sorted(set(ct_site)):
	vecsbysite[site] = features_EB[ct_site==site]

specsbysite = {}
for site in sorted(set(ct_site)):
	specsbysite[site] = species[ct_site==site]

vecsbymgmt = {}
for site in sorted(set(mgmt)):
	vecsbymgmt[site] = features_EB[mgmt==site]

specsbymgmt = {}
for site in sorted(set(mgmt)):
	specsbymgmt[site] = species[mgmt==site]

plot = plt.scatter(u[:,0], u[:,1])
plt.title('UMAP embedding CT images');

#plot - all images coloured by species
sns.set_theme(style="white")
sns.relplot(x=u[:,0], y= u[:,1], hue=species, alpha=.2, palette="muted",
            height=10).set(title = 'UMAP embedding coloured by species')
plt.savefig('output/figs/allimgs/umap_species_Emb.png', dpi='figure')

#plot - all coloured by ct_site
sns.set_theme(style="white")
sns.relplot(x=u[:,0], y= u[:,1], hue=ct_site, alpha=.2, palette="mako",
            height=10).set(title = 'UMAP embedding coloured by site')
plt.savefig('output/figs/allimgs/umap_sites_Emb.png', dpi='figure')

#plots coloured by management zone
sns.set_theme(style="white")
sns.relplot(x=u[:,0], y= u[:,1], hue=mgmt, alpha=.2, palette="mako",
            height=10).set(title = 'UMAP embedding coloured by management zone')
plt.savefig('output/figs/allimgs/umap_mgmt_Emb.png', dpi='figure')

#plots coloured by time of day
sns.set_theme(style="white")
sns.relplot(x=u[:,0], y= u[:,1], hue=time_hour, alpha=.2, palette="rocket",
            height=10).set(title = 'UMAP embedding coloured by time of day')
plt.savefig('output/figs/allimgs/umap_Emb.png', dpi='figure')

print('Plots saved!')

k_EB = pd.DataFrame(sorted(set(ct_site)), columns = ['site'])
k_EB['numimgs'] = pd.NaT
k_EB['nspecies'] = pd.NaT
k_EB['OK_2048'] = pd.NaT
k_EB['OK_512'] = pd.NaT
k_EB['OK_128'] = pd.NaT
k_EB['OK_32'] = pd.NaT
k_EB['OK_8'] = pd.NaT
k_EB['OK_2'] = pd.NaT
dimensions = [2048,512,128,32,8,2]
kcomp = k_EB #just for the testing part
kcomp = kcomp.reset_index()  # make sure indexes pair with number of row


dimensions = [2048,512,128,32,8,2]
kcomp = k_EB #just for the testing part
kcomp = kcomp.reset_index()  # make sure indexes pair with number of row

for index,row in kcomp.iterrows():
	kmax = 15
	site = row['site']
	x = vecsbysite[site]
	y = specsbysite[site]
	kcomp['numimgs'][index] = len(x)
	kcomp['nspecies'][index] = len(set(y))
	#finding optimal value of k using the silhouette coefficient
	#if number of samples is less than kmax, change optimK
	for dim in dimensions:
		sil = []
		print(f'Finding optimal k for site {site} with {dim} dimensions')
		if len(x) > 5:
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
k_EB = kcomp

k_EB.to_csv('output/kmeans_EB_dims.csv', index=False)
