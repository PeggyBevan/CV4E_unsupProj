'''
Taking in feature vectors from EmbNet model and visualising
Running Kmeans clustering on features
Finding rank score
Doing the same for day time images
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
#must be unsupProj dir
from functions import findk


#cd ../../
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

#--> if doing day time images, skip to next section

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





#####-------repeat for daytime only------#######

print('removing nocturnal images')
dayfeatures = []
dayhour = []
dayspecies = []
daymgmt = []
daysite = []
for i, v in enumerate(features_EB):
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


k_EB_day = pd.DataFrame(sorted(set(daysite)), columns = ['site'])
k_EB_day['numimgs'] = pd.NaT
k_EB_day['nspecies'] = pd.NaT
k_EB_day['OK_2048'] = pd.NaT

dimensions = [2048]
test = findk(kcomp = k_EB_day, dimensions = dimensions, vecsbysite = vecsbysite, specsbysite = specsbysite)

k_EB_day.to_csv('output/kmeans_EB_day.csv', index=False)


#HBDSCAN on day time pics only.
#testing on one site - BZ08
x = vecsbysite['BZ08']
y = specsbysite['BZ08']
lowd = umap.UMAP().fit_transform(x)
#we can plot this
sns.set_theme(style="white")
sns.relplot(x=lowd[:,0], y= lowd[:,1], hue=y, alpha=.4, palette="muted",
            height=10).set(title = 'EmbNet embeddings daytime coloured by species, BZ08')
plt.savefig('output/figs/BySite/umap_EBd_BZ08.png', dpi='figure')


x = vecsbysite['NP10']
y = specsbysite['NP10']
lowd = umap.UMAP().fit_transform(x)
#we can plot this
sns.set_theme(style="white")
sns.relplot(x=lowd[:,0], y= lowd[:,1], hue=y, alpha=.4, palette="muted",
            height=10).set(title = 'EmbNet embeddings daytime coloured by species, NP10')


#With HDBSCAN, you give a minimum number of clusters only, and the alogorithm sets
# finds the optimal number, so no need to search for best number.
#You just set min number of clusters and min size of clusters.

#funcion to make 2D representation from any site
#relies on vecsbysite and specsbysite existing


def lowD_site(site):
	x = vecsbysite[site]
	y = specsbysite[site]
	#its important that UMAP results are reproducible e.g. randomstate
	lowd = umap.UMAP(init = 'random').fit_transform(x)
	return(lowd)

lowd = lowD_site('BZ08')

clusterer = hdbscan.HDBSCAN(min_cluster_size=30, min_samples = 15)
clusterer.fit(lowd)
#look at labels
clusterer.labels_
#check max number of labels:
clusterer.labels_.max()
#plot this:
sns.relplot(x=lowd[:,0], y= lowd[:,1], hue=clusterer.labels_, alpha=.4, palette="muted")

##NOtes 13/09
# I found that playing around with the min cluster size changes the
# predicted number of clusters. 
# the best so far (from only looking at one site) seems to be 
#min cluster size = 30, min_samples = 15. 
#to show that I am not playing around, I need to do this a couple of times
#with diff parameters and see what happens. 
#For now, run clustering on every site to see how it looks overall. 
#Then you can think about how to adjust paramters. 
#But would be good to think about making function where parameters can be easily changed

#First, lets see how this looks on another site.
lowd = lowD_site('NP10')

clusterer = hdbscan.HDBSCAN(min_cluster_size=30, min_samples = 15)
clusterer.fit(lowd)
#check max number of labels:
clusterer.labels_.max()

#plot this:
sns.relplot(x=lowd[:,0], y= lowd[:,1], hue=clusterer.labels_, alpha=.4, palette="muted")
sns.relplot(x=lowd[:,0], y= lowd[:,1], hue=specsbysite['NP10'], alpha=.4, palette="muted")



#testing changing n_neigbours to 30 and min dist to 0
clusterable_embedding = umap.UMAP(
    n_neighbors=30,
    min_dist=0.0,
    n_components=512,
    random_state=42,
).fit_transform(x)

labels = hdbscan.HDBSCAN(
    min_samples=10,
    min_cluster_size=2,
).fit_predict(clusterable_embedding)

#if you increase/decrease min_cluster size, min samples scales with it. 
# so set independently 

#how to test this on all sites, on all parameters
#set site, n components, hdb min samples, min cluster size
#for now, don't play around too much with UMAP - the clustering parameters
#seem to be more important. I tried with a few dimensions and it didn't seem
#to make a difference, except when I did no reduction (2048) and everyting
#was classed as noise!



##RUNNING HDBSCAN ON ALL SITES
#set clusterer before loop
clusterer = hdbscan.HDBSCAN(min_cluster_size=30, min_samples = 15)

df = pd.DataFrame(sorted(set(daysite)), columns = ['site'])
df['nspecies'] = pd.NaT
df['nimgs'] = pd.NaT
df['labels_30_15'] = pd.NaT
df['noise_30_15'] = pd.NaT



def hdbfinder(df, clusterer, ClusterCol, NoiseCol ):
	for index, row in df.iterrows():
		site = row['site']
		x = vecsbysite[site]
		y = specsbysite[site]
		print(f'Performing clustering for site {site}')
		#get some ground-truth metrics from data
		df['nimgs'][index] = len(x)
		df['nspecies'][index] = len(set(y))
		if len(x) > 5:
			#reduce vectors to 2 dims
			lowd = umap.UMAP(random_state = 42).fit_transform(x)
			clusterer.fit(lowd)
			#check max number of labels (+1 to include 0)
			df[ClusterCol][index] = clusterer.labels_.max()+1
			#number of points that can't be assigned (label = -1)
			df[NoiseCol][index] = len(clusterer.labels_[clusterer.labels_<0])
		else:
			print(f'{site} has low sample size, moving on')
			df[ClusterCol][index] = pd.NaT
			df[NoiseCol][index] = pd.NaT

sns.histplot(df.labels_30_15.dropna())
plt.savefig('output/figs/HDBSCAN/EBd_30_15.png', dpi = 'figure')


clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
df['labels_5'] = pd.NaT
df['noise_5'] = pd.NaT
hdbfinder(df, clusterer, 'labels_5', 'noise_5')
sns.histplot(df.labels_5.dropna())
plt.savefig('output/figs/HDBSCAN/EBd_30_15.png', dpi = 'figure')


clusterer = hdbscan.HDBSCAN(min_cluster_size=15)
df['labels_15'] = pd.NaT
df['noise_15'] = pd.NaT
hdbfinder(df, clusterer, 'labels_15', 'noise_15')
sns.histplot(df.labels_15.dropna())
plt.savefig('output/figs/HDBSCAN/EBd_15.png', dpi = 'figure')


clusterer = hdbscan.HDBSCAN(min_cluster_size=30)
df['labels_30'] = pd.NaT
df['noise_30'] = pd.NaT
hdbfinder(df, clusterer, 'labels_30', 'noise_30')
sns.histplot(df.labels_30.dropna())
plt.savefig('output/figs/HDBSCAN/EBd_30.png', dpi = 'figure')

clusterer = hdbscan.HDBSCAN(min_cluster_size=60)
df['labels_60'] = pd.NaT
df['noise_60'] = pd.NaT
hdbfinder(df, clusterer, 'labels_60', 'noise_60')
sns.histplot(df.labels_60.dropna())
plt.savefig('output/figs/HDBSCAN/EBd_60.png', dpi = 'figure')

clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples = 5)
df['labels_15_5'] = pd.NaT
df['noise_15_5'] = pd.NaT
hdbfinder(df, clusterer, 'labels_15_5', 'noise_15_5')
sns.histplot(df.labels_15_5.dropna())
plt.savefig('output/figs/HDBSCAN/EBd_15_5.png', dpi = 'figure')

clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples = 15)
df['labels_15_15'] = pd.NaT
df['noise_15_15'] = pd.NaT
hdbfinder(df, clusterer, 'labels_15_15', 'noise_15_15')
sns.histplot(df.labels_15_15.dropna())
plt.savefig('output/figs/HDBSCAN/EBd_15_15.png', dpi = 'figure')

clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples = 30)
df['labels_15_30'] = pd.NaT
df['noise_15_30'] = pd.NaT
hdbfinder(df, clusterer, 'labels_15_30', 'noise_15_30')
sns.histplot(df.labels_15_30.dropna())
plt.savefig('output/figs/HDBSCAN/EBd_15_30.png', dpi = 'figure')


#as min cluster increases, so does min samples.
#lets see what happens when we reduce min samples and increase cluster size
clusterer = hdbscan.HDBSCAN(min_cluster_size=30, min_samples = 5)
df['labels_30_05'] = pd.NaT
df['noise_30_05'] = pd.NaT
hdbfinder(df, clusterer, 'labels_30_05', 'noise_30_05')
sns.histplot(df.labels_30_05.dropna())
plt.savefig('output/figs/HDBSCAN/EBd_30_05.png', dpi = 'figure')

clusterer = hdbscan.HDBSCAN(min_cluster_size=60, min_samples = 5)
df['labels_60_05'] = pd.NaT
df['noise_60_05'] = pd.NaT
hdbfinder(df, clusterer, 'labels_60_05', 'noise_60_05')
sns.histplot(df.labels_30_05.dropna())
plt.savefig('output/figs/HDBSCAN/EBd_30_05.png', dpi = 'figure')

#parameters to test:
#min cluster size: 5,15,30,60
#min_samples: 5,15,30,60

#just change cluster size, see how that makes a difference


#UMAP:
#n_neighbors: 15,30,60



