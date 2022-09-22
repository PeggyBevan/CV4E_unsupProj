'''
Clustering by site - kmeans and 
Creating Metrics for model performance and visualising.
Peggy Bevan 2022
'''

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import umap
import hdbscan
import sklearn.cluster as cluster
from sklearn import metrics, datasets
import colorcet as cc

from functions import findk

#function for pop-up figure
%matplotlib auto 

####------Load Data----######
#read in numpy arrays - output from predict.py
print('reading in feature vectors')
features_PN = np.load('output/PegNet/PegNet_fvect_norm.npy')
imgs_PN = np.load('output/PegNet/PegNet_imgvect_norm.npy')

features_EB = np.load('output/EmbNet/Emb_fvect.npy')
imgs_EB = np.load('output/EmbNet/Emb_imgvect.npy')

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

print('Organising data by site')
vecsbysitePN = {}
for site in sorted(set(ct_site)):
	vecsbysitePN[site] = features_PN[ct_site==site]

vecsbysiteEB = {}
for site in sorted(set(ct_site)):
	vecsbysiteEB[site] = features_EB[ct_site==site]

specsbysite = {}
for site in sorted(set(ct_site)):
	specsbysite[site] = species[ct_site==site]

#reduce dims on all features, before subsetting
fit = umap.UMAP(random_state = 42, min_dist = 0.0)
u = fit.fit_transform(features_PN)

UMAPbysitePN = {}
for site in sorted(set(ct_site)):
	UMAPbysitePN[site] = u[ct_site==site]

fit = umap.UMAP(random_state = 42, min_dist = 0.0)
u = fit.fit_transform(features_EB)

UMAPbysiteEB = {}
for site in sorted(set(ct_site)):
	UMAPbysiteEB[site] = u[ct_site==site]

####------K-means clustering---######

#testing stability of k predictions:
#everytime you reduce features using UMAP, the output is based on a random sequence
#however, hopefully the points which are similar to each other will come out the same.
#if optimal k comes out the same each time, then the analysis is stable.





for i in range(1,5):
	x = vecsbysite['BZ03']
	kmax = 30
	dim = 512 #need to reduce dimensions
	print('Compressing to 512 dims')
	embedding = umap.UMAP(n_components=dim).fit_transform(x)
	print('Compressing successful')
	sil = []
	for k in range(2, kmax+1):
  		kmeans = cluster.KMeans(n_clusters = k).fit(embedding)
  		labels = kmeans.labels_ #underscore is needed
  		sil.append(metrics.silhouette_score(embedding, labels, metric = 'euclidean'))
	#find index of max silhouette and add 2 (0 = 2)
	optimk = sil.index(max(sil))+2
	print(f'{i} = {optimk}')
'''
#output:
1 = 2
2 = 2
3 = 2
4 = 2
5 = 2
'''

#Plot BZ03
x = vecsbysitePN['BZ03']
xeb = vecsbysiteEB['BZ03']
y = specsbysite['BZ03']
embedding = umap.UMAP(random_state = 42, min_dist = 0.0).fit_transform(x)

f, ax = plt.subplots(figsize=(8, 8))
sns.set_style("white")
gpalette = sns.color_palette(cc.glasbey_bw, n_colors=25)
g = sns.scatterplot(x=embedding[:,0], y= embedding[:,1], hue=y, alpha=.6, s = 3, ax = ax)
sns.move_legend(g, "upper left", bbox_to_anchor=(1,1), frameon = False, fontsize = 10)
plt.title('BZ03 Embeddings', fontsize = 12)
plt.tight_layout()

#notes 22/09/22 - Comparing visulisations for PN and EB do not look very different at all.

#does compressing on entire data vs subset make a difference?
x = UMAPbysitePN['BZ03']
xeb = UMAPbysiteEB['BZ03']
y = specsbysite['BZ03']
embedding = umap.UMAP(random_state = 42, min_dist = 0.0).fit_transform(x)

f, ax = plt.subplots(figsize=(8, 8))
sns.set_style("white")
g = sns.scatterplot(x=embedding[:,0], y= embedding[:,1], hue=y, alpha=.6, s = 3, ax = ax)
sns.move_legend(g, "upper left", bbox_to_anchor=(1,1), frameon = False, fontsize = 10)
plt.title('BZ03 Embeddings, compressed pre subset', fontsize = 12)
plt.tight_layout()

#This looks VERY different!
#For some reason, performing UMAP twice on the data allows it to cluster way better.
#It would be interesting to play with dimensions and see what happens

#now, lets try k-means with this new method
for i in range(1,5):
	x = UMAPbysitePN['BZ03']
	kmax = 10
	sil = []
	calh = []
	print('Compressing to 2 dimensions')
	embedding = umap.UMAP(random_state = 42, min_dist = 0.0).fit_transform(x)
	print('Compression successful')
	for k in range(2, kmax+1):
  		kmeans = cluster.KMeans(n_clusters = k, random_state = 42).fit(embedding)
  		labels = kmeans.labels_ #underscore is needed
  		sil.append(metrics.silhouette_score(x, labels, metric = 'euclidean'))
  		calh.append(metrics.calinski_harabasz_score(x, labels))
	#find index of max silhouette and add 2 (0 = 2)
	optimk = sil.index(max(sil))+2
	optimk2 = calh.index(max(calh))+2
	print(f'{i}:Silhoutte score n clusters = {optimk}\nCalinski Harabasz score n clusters = {optimk2}')

#Notes - 22/09 - The C-H metric continually predicts max k, whatever that is set to.
# So seems like it is overfitting quite a lot. It is also inconsistent, where silhouette always chooses the
# same optimal K. 
#Without random state - optimal k is very inconsistent.
#With random state - the number is still jumping around a lot - between two numbers. 
#There needs to be random state set in the kmeans clustering execution as well!!
#Tried with NP10 as well - it just went to the maximum number of clusters...
#Using x or embedding in the metrics: doesn't seem to make any diffence to 
#output 

#possible parameters
#clusterable_embedding = umap.UMAP(
    #n_neighbors=30,
    #min_dist=0.0,
    #n_components=4,
    #random_state=42,
#).fit_transform(x)


#finding optimal K for each site, testing dimensionality
k_PN = pd.DataFrame(sorted(set(ct_site)), columns = ['site'])
k_PN['numimgs'] = pd.NaT
k_PN['nspecies'] = pd.NaT
k_PN['OK_2048'] = pd.NaT
k_PN['OK_512'] = pd.NaT
k_PN['OK_128'] = pd.NaT
k_PN['OK_32'] = pd.NaT
k_PN['OK_8'] = pd.NaT
k_PN['OK_2'] = pd.NaT


dimensions = [2048,512,128,32,8,2]
kcomp = k_PN #just for the testing part
kcomp = kcomp.reset_index()  # make sure indexes pair with number of row

#kcomp is a dataframe with numimgs, nspecies, and a column called 'OK_dim' for each dimension
#dimensions is a list
#vecs & specs by site is a dictionary with list of vectors and labels in the same order
def findk(kcomp, dimensions, vecsbysite, specsbysite, kmax=15):
	for index,row in kcomp.iterrows():
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

k_EB = kcomp
k_PN = kcomp

#save
k_PN.to_csv('output/kmeans_PN_dims.csv', index=False)

#Omi model

#finding optimal K for each site, testing dimensionality
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


#kmeans on day time pictures, only 2048 dimensions

np.array(dayspecies)
np.array(daysite)

vecsbysite = {}
for site in sorted(set(daysite)):
	vecsbysite[site] = dayfeatures[daysite==site]

specsbysite = {}
for site in sorted(set(daysite)):
	specsbysite[site] = dayspecies[daysite==site]

k_PN_day = pd.DataFrame(sorted(set(ct_site)), columns = ['site'])
k_PN_day['numimgs'] = pd.NaT
k_PN_day['nspecies'] = pd.NaT
k_PN_day['OK_2048'] = pd.NaT



kcomp = k_PN_day
kcomp = kcomp.reset_index()
for index,row in kcomp.iterrows():
	kmax = 15
	site = row['site']
	x = vecsbysite[site]
	y = specsbysite[site]
	kcomp['numimgs'][index] = len(x)
	kcomp['nspecies'][index] = len(set(y))
	if len(x) > 5:
		embedding = umap.UMAP(init = 'random', n_components=dim).fit_transform(x)




#interpreting and plotting results from clustering analysis
#PegNet
kmeans_PN = pd.read_csv('output/PegNet/kmeans_PN_dims.csv')
kmeans_EB = pd.read_csv('output/EmbNet/kmeans_EB_dims.csv')
kmeans_PNd = pd.read_csv('output/PegNet/kmeans_PN_day.csv')
kmeans_EBd = pd.read_csv('output/PegNet/kmeans_EB_day.csv')

kmeans = kmeans_PN #depending on which you are using
kmeans = kmeans_EB
kmeans = kmeans_PNd
kmeans = kmeans_EBd

#plot histograms
sns.histplot(kmeans.nspecies).set(title = 'Number of species by camera trap site')
sns.histplot(kmeans.OK_2048).set(title = 'Optimal number of clusters by camera trap site')

plt.savefig('output/figs/allimgs/specieshist.png', dpi='figure')
plt.savefig('output/figs/allimgs/clusterhist_2048.png', dpi='figure')

sns.histplot(kmeans.OK_512).set(title = 'Optimal number of clusters by camera trap site')
sns.histplot(kmeans.OK_128).set(title = 'Optimal number of clusters by camera trap site')
sns.histplot(kmeans.OK_32).set(title = 'Optimal number of clusters by camera trap site')
sns.histplot(kmeans.OK_8).set(title = 'Optimal number of clusters by camera trap site')
sns.histplot(kmeans.OK_2).set(title = 'Optimal number of clusters by camera trap site')
#plot scatter plot

#error plot
kmeans['error_2048'] = (kmeans.OK_2048 - kmeans.nspecies)
kmeans['error_512'] = (kmeans.nspecies - kmeans.OK_512)
kmeans['error_128'] = (kmeans.nspecies - kmeans.OK_128)
kmeans['error_32'] = (kmeans.nspecies - kmeans.OK_32)
kmeans['error_8'] = (kmeans.nspecies - kmeans.OK_8)
kmeans['error_2'] = (kmeans.nspecies - kmeans.OK_2)
sns.histplot(kmeans.error_2048).set(title = 'Residual error, PegNet50 embeddings with 2048 dimensions')
sns.histplot(kmeans.error_512).set(title = 'Residual error, PegNet50 embeddings with 512 dimensions')
sns.histplot(kmeans.error_128).set(title = 'Residual error, PegNet50 embeddings with 128 dimensions')
sns.histplot(kmeans.error_32).set(title = 'Residual error, PegNet50 embeddings with 32 dimensions')
sns.histplot(kmeans.error_8).set(title = 'Residual error, PegNet50 embeddings with 8 dimensions')
sns.histplot(kmeans.error_2).set(title = 'Residual error, PegNet50 embeddings with 2 dimensions')

plt.savefig('output/figs/allimgs/errorhist_EB_2048.png', dpi='figure')
###calculating ranks between species numbers and clusters

#if the sites with high species richness are also the sites of high cluster numbers, then there is a good similarity


def rankscore(ground_truth, prediction, data):
	x = len(ground_truth)
	totalop = ((x**2)-x)/2 #total number of combinations, less the diagonal and bottom half of matrix
	score = 0
	for i, row in data.iterrows():
		for j in range(i+1, len(data)):
			if ground_truth[i] > ground_truth[j]:
				if prediction[i] > prediction[j]:
					score += 1
			if ground_truth[i] < ground_truth[j]:
				if prediction[i] < prediction[j]:
					score += 1
			if ground_truth[i] == ground_truth[j]:
				if prediction[i] == prediction[j]:
					score += 1 
	rankscore = score/totalop
	print(f'Rank score = {rankscore}')
	return(rankscore)


rank_PN2048 = rankscore(kmeans['nspecies'], kmeans['OK_2048'], kmeans)
rank_PN512 = rankscore(kmeans['nspecies'], kmeans['OK_512'], kmeans)
rank_PN128 = rankscore(kmeans['nspecies'], kmeans['OK_128'], kmeans)
rank_PN32 = rankscore(kmeans['nspecies'], kmeans['OK_32'], kmeans)
rank_PN08 = rankscore(kmeans['nspecies'], kmeans['OK_8'], kmeans)
rank_PN02 = rankscore(kmeans['nspecies'], kmeans['OK_2'], kmeans)

rank_EB2048 = rankscore(kmeans['nspecies'], kmeans['OK_2048'], kmeans)
rank_EB512 = rankscore(kmeans['nspecies'], kmeans['OK_512'], kmeans)
rank_EB128 = rankscore(kmeans['nspecies'], kmeans['OK_128'], kmeans)
rank_EB32 = rankscore(kmeans['nspecies'], kmeans['OK_32'], kmeans)
rank_EB08 = rankscore(kmeans['nspecies'], kmeans['OK_8'], kmeans)
rank_EB02 = rankscore(kmeans['nspecies'], kmeans['OK_2'], kmeans)

##Visialising EmbNet results
kmeans_EB = pd.read_csv('output/EmbNet/kmeans_Emb_dims.csv')

kmeans = kmeans_EB
#plot histograms
sns.histplot(kmeans.OK_2048).set(title = 'Optimal number of clusters by camera trap site')

plt.savefig('output/figs/allimgs/EB_2048.png', dpi='figure')

sns.histplot(kmeans.OK_512).set(title = 'Optimal number of clusters by camera trap site')
plt.savefig('output/figs/allimgs/EB_512.png', dpi='figure')

sns.histplot(kmeans.OK_128).set(title = 'Optimal number of clusters by camera trap site')
plt.savefig('output/figs/allimgs/EB_128.png', dpi='figure')

sns.histplot(kmeans.OK_32).set(title = 'Optimal number of clusters by camera trap site')
plt.savefig('output/figs/allimgs/EB_32.png', dpi='figure')

sns.histplot(kmeans.OK_8).set(title = 'Optimal number of clusters by camera trap site')
plt.savefig('output/figs/allimgs/EB_08.png', dpi='figure')

sns.histplot(kmeans.OK_2).set(title = 'Optimal number of clusters by camera trap site')
plt.savefig('output/figs/allimgs/EB_02.png', dpi='figure')


####-----HDBSCAN----######

clusterer = hdbscan.HDBSCAN(min_cluster_size=30, min_samples = 15)
x = UMAPbysitePN['BZ03']
embedding = umap.UMAP(random_state = 42, min_dist = 0.0).fit_transform(x)
clusterer.fit(embedding)
#look at labels
clusterer.labels_
#check max number of labels:
clusterer.labels_.max()
#plot this:
sns.relplot(x=embedding[:,0], y= embedding[:,1], hue=clusterer.labels_, alpha=.4, palette="muted")

clusterer = hdbscan.HDBSCAN(min_cluster_size=30, min_samples = 15)

df = pd.DataFrame(sorted(set(ct_site)), columns = ['site'])
df['nspecies'] = pd.NaT
df['nimgs'] = pd.NaT
df['labels_30_15'] = pd.NaT
df['noise_30_15'] = pd.NaT

def hdbfinder(df, clusterer, ClusterCol, NoiseCol ):
	for index, row in df.iterrows():
		site = row['site']
		x = UMAPbysitePN[site]
		y = specsbysite[site]
		print(f'Performing clustering for site {site}')
		#get some ground-truth metrics from data
		df['nimgs'][index] = len(x)
		df['nspecies'][index] = len(set(y))
		if len(x) > 5:
			#reduce vectors to 2 dims
			lowd = umap.UMAP(random_state = 42, min_dist = 0.0).fit_transform(x)
			clusterer.fit(lowd)
			#check max number of labels (+1 to include 0)
			df[ClusterCol][index] = clusterer.labels_.max()+1
			#number of points that can't be assigned (label = -1)
			df[NoiseCol][index] = len(clusterer.labels_[clusterer.labels_<0])
		else:
			print(f'{site} has low sample size, moving on')
			df[ClusterCol][index] = pd.NaT
			df[NoiseCol][index] = pd.NaT

hdbfinder(df, clusterer, 'labels_30_15', 'noise_30_15')
sns.histplot(df.labels_30_15).dropna()
sns.scatterplot(x = df.nspecies, y = df.labels_30_15)

#I think that min cluster size is impacting sites with low sample size, so
# I want to see what happens when I reduce this number but keep min samples the same

df['labels_05_15'] = pd.NaT
df['noise_05_15'] = pd.NaT
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples = 15)
hdbfinder(df, clusterer, 'labels_05_15', 'noise_05_15')
sns.histplot(df.labels_05_15).dropna()
sns.scatterplot(x = df.nspecies, y = df.labels_05_15)



