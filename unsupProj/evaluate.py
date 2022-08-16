'''
	Taking in feature vectors from model and visualising using UMAP
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
import sklearn.cluster as cluster
from collections import defaultdict
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn import datasets
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

#%matplotlib inline


#read in numpy arrays - output from predict.py
print('reading in feature vectors')
features = np.load('output/featurevector.npy')
imgs = np.load('output/imgpathvector.npy')

#add information about camera trap location and species
excel = 'data/nepal_cropsmeta_PB.csv'
meta = pd.read_csv(excel)

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

##split data by camera trap site
print('Organising data by site')
vecsbysite = {}
for site in set(ct_site):
	vecsbysite[site] = features[ct_site==site]

specsbysite = {}
for site in set(ct_site):
	specsbysite[site] = species[ct_site==site]

vecsbymgmt = {}
for site in set(mgmt):
	vecsbymgmt[site] = features[mgmt==site]

specsbymgmt = {}
for site in set(mgmt):
	specsbymgmt[site] = species[mgmt==site]

'''#save these
json.dumps(vecsbysite)
json.dumps(specsbysite)
json.dumps(vecsbymgmt)
json.dumps(specsbymgmt)
'''	

#Visualising entire dataset
#create umap object
print('Plotting UMAP embeddings for entire dataset')
fit = umap.UMAP()
u = fit.fit_transform(features)


#initial plot, no colours
plot = plt.scatter(u[:,0], u[:,1])
plt.title('UMAP embedding CT images');

#plot - all images coloured by species
sns.set_theme(style="white")
sns.relplot(x=u[:,0], y= u[:,1], hue=species, alpha=.2, palette="muted",
            height=10).set(title = 'UMAP embedding coloured by species')
plt.savefig('output/figs/allimgs/umap_species.png', dpi='figure', )

#plot - all coloured by ct_site
sns.set_theme(style="white")
sns.relplot(x=u[:,0], y= u[:,1], hue=ct_site, alpha=.2, palette="muted",
            height=10).set(title = 'UMAP embedding coloured by cite')
plt.savefig('output/figs/allimgs/umap_sites.png', dpi='figure', )

#plots coloured by management zone
sns.set_theme(style="white")
sns.relplot(x=u[:,0], y= u[:,1], hue=mgmt, alpha=.2, palette="muted",
            height=10).set(title = 'UMAP embedding coloured by management zone')
plt.savefig('output/figs/allimgs/umap_mgmt.png', dpi='figure')

#plots coloured by time of day
sns.set_theme(style="white")
sns.relplot(x=u[:,0], y= u[:,1], hue=time_hour, alpha=.2, palette="muted",
            height=10).set(title = 'UMAP embedding coloured by time of day')
plt.savefig('output/figs/allimgs/umap_hour.png', dpi='figure')

print('Plots saved!')
'''
#plotting for one ct site
fit = umap.UMAP()
%time BZ01 = fit.fit_transform(vecsbysite['BZ01'])
plt.scatter(BZ01[:,0], BZ01[:,1])

#index species numbers to colour by
BZ01s = []
for i in range(len(specsbysite['BZ01'])):
	a = zip_specvals[specsbysite['BZ01'][i]]
	BZ01s.append(a)
plot = plt.scatter(BZ01[:,0], BZ01[:,1], c = BZ01s)
plt.title('UMAP embedding: BZ01')
legend1 = plt.legend(*plot.legend_elements(),
                    loc="top right", title="Classes")
plt.add_artist(legend1)


fit = umap.UMAP()
%time NP09= fit.fit_transform(vecsbysite['NP09'])

NP09s = []
for i in range(len(specsbysite['NP09'])):
	a = zip_specvals[specsbysite['NP09'][i]]
	NP09s.append(a)
plot = plt.scatter(NP09[:,0], NP09[:,1], c = NP09s)
plt.title('UMAP embedding: NP09')
legend1 = ax.legend(*plot.legend_elements(),
                    loc="lower left", title="Classes")
plt.legend(legend1)

fit = umap.UMAP()
%time OBZ04 = fit.fit_transform(vecsbysite['OBZ04'])

OBZ04s = []
for i in range(len(specsbysite['OBZ04'])):
	a = zip_specvals[specsbysite['OBZ04'][i]]
	OBZ04s.append(a)
plot = plt.scatter(OBZ04[:,0], OBZ04[:,1], c = OBZ04s)
plt.title('UMAP embedding: OBZ04')
legend1 = plt.legend(*plot.legend_elements(),
                    loc="lower left", title="Classes")
plt.legend(legend1)
'''


##make a plot for each camera trap site
#loop method through ct_site

#find optimal k for each site
#the max number of species seen at any site is 14.
kmax = 30
kcomp = pd.DataFrame(set(ct_site), columns = ['site'])
kcomp['numimgs'] = pd.NaT
kcomp['nspecies'] = pd.NaT
kcomp['optimK'] = pd.NaT


kcomp = kcomp.reset_index()  # make sure indexes pair with number of rows
print('Beginning k-means clustering')
for index,row in kcomp.iterrows():
	sil = []
	site = row['site']
	x = vecsbysite[site]
	y = specsbysite[site]
	kcomp['numimgs'][index] = len(x)
	kcomp['nspecies'][index] = len(set(y))
	print(f'Finding optimal k for site {site}')
	#finding optimal value of k using the silhouette coefficient
	for k in range(2, kmax+1):
  		kmeans = cluster.KMeans(n_clusters = k).fit(x)
  		labels = kmeans.labels_ #underscore is needed
  		sil.append(metrics.silhouette_score(x, labels, metric = 'euclidean'))
  		#find index of max silhouette and add 2 (0 = 2)
  		optimk = sil.index(max(sil))+2
  		kcomp['optimK'][index] = optimk

#save
kcomp.to_csv('output/kmeans_nspecies_comp.csv', index=False)


