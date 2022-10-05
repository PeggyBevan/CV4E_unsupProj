'''
#input variables
#df - a dataframe with site, numimgs, numspecies and columns for each clustering methods
for kmeans - 'OK' with chosen dimension. - dimensions not needed if clustering happening before?
for HDB - HD_30_15 with clusterer settings. HD_noise - for number in noise

dimensions - i think not needed if UMAP happens before

vecsbysite - will be Umap by site - dictionary with umap representations listed by site
speces by site - same but with species listed by site
kmax - maximum number of clusters.

df = pd.DataFrame(sorted(set(ct_site)), columns = ['site'])
df['nimgs'] = pd.NaT
df['nspecies'] = pd.NaT
df['OK'] = pd.NaT
df['HDB'] = pd.NaT
df['HDB_noise'] = pd.NaT

clusterer = hdbscan.HDBSCAN(min_cluster_size=30, min_samples = 15)

'''
def findclusters(df, UMAPbysite, specsbysite, kmax=15):
	for index,row in df.iterrows():
		site = row['site']
		x = UMAPbysite[site]
		y = specsbysite[site]
		df['nimgs'][index] = len(x)
		df['nspecies'][index] = len(set(y))
		#finding optimal value of k using the silhouette coefficient
		#if number of samples is less than kmax, change optimK
		sil = []
		print(f'Finding optimal k for site {site}')
		if len(x) > 5: #don't bother clustering on sites with less than 5 examples
			if len(x) <= kmax:
				kmax = len(x)
				for k in range(2, kmax):
					print(f'k = {k}')
					kmeans = cluster.KMeans(n_clusters = k).fit(x)
					labels = kmeans.labels_ #underscore is needed
					sil.append(metrics.silhouette_score(x, labels, metric = 'euclidean'))
					#find index of max silhouette and add 2 (0 = 2)	
				optimk = sil.index(max(sil))+2
			else:
				for k in range(2, kmax+1):
					print(f'k = {k}')
					kmeans = cluster.KMeans(n_clusters = k).fit(x)
					labels = kmeans.labels_ #underscore is needed
					sil.append(metrics.silhouette_score(x, labels, metric = 'euclidean'))
					#find index of max silhouette and add 2 (0 = 2)
				optimk = sil.index(max(sil))+2
			#perform hdbscan
			print(f'Performing HDBSCAN clustering for site {site}')
			clusterer.fit(x)
			#check max number of labels (+1 to include 0)
			HDB_n = clusterer.labels_.max()+1
			#number of points that can't be assigned (label = -1)
			HDB_noise = len(clusterer.labels_[clusterer.labels_<0])
		else:
			print(f'{site} has low sample size, moving on')
			optimk = 'NA'
			HDB_n = 'NA'
			HDB_noise = 'NA'
		df['OK'][index] = optimk
		df['HDB'][index] = HDB_n
		df['HDB_noise'][index] = HDB_noise
		print(f'Optimal cluster number (kmeans) = {optimk}')
		print(f'Optimal cluster number (HDBSCAN) = {HDB_n}')

