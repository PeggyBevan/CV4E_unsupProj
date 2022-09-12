'''
Creating Metrics for model performance and visualising.
Peggy Bevan 2022
'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#function for pop-up figure
%matplotlib auto 

#interpreting and plotting results from clustering analysis
#PegNet
kmeans_PN = pd.read_csv('output/PegNet/kmeans_PN_dims.csv')
kmeans_EB = pd.read_csv('output/EmbNet/kmeans_EB_dims.csv')
kmeans_PNd = pd.read_csv('output/PegNet/kmeans_PN_day.csv')

kmeans = kmeans_PN #depending on which you are using
kmeans = kmeans_EB
kmeans = kmeans_PNd

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




