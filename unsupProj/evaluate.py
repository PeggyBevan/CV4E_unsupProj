'''
	Taking in feature vectors from model and visualising using UMAP - SWAV
	Peggy Bevan Aug 2022
'''
#if numpy files need importing, in cmd line:
#scp -i ~/Downloads/PegNetVM_key.pem "pegbev@20.228.65.187:/home/pegbev/output/PegNet_fvect_norm.npy" "/Users/peggybevan/OneDrive/My Documents/PhD/CameraTraps/CV4E/output/"
#scp -i ~/Downloads/PegNetVM_key.pem "pegbev@20.228.65.187:/home/pegbev/output/PegNet_imgvect_norm.npy" "/Users/peggybevan/OneDrive/My Documents/PhD/CameraTraps/CV4E/output/"
#scp -i ~/Downloads/PegNetVM_key.pem "pegbev@20.228.65.187:/home/pegbev/output/Swav_fvect.npy" "/Users/peggybevan/OneDrive/My Documents/PhD/CameraTraps/CV4E/output/"
#scp -i ~/Downloads/PegNetVM_key.pem "pegbev@20.228.65.187:/home/pegbev/output/Swav_imgvect.npy" "/Users/peggybevan/OneDrive/My Documents/PhD/CameraTraps/CV4E/output/"


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
import datashader as ds

%matplotlib auto #display plots


#read in numpy arrays - output from predict.py
print('reading in feature vectors')
features_PN = np.load('output/PegNet/PegNet_fvect_norm.npy')
imgs_PN = np.load('output/PegNet/PegNet_imgvect_norm.npy')

features_Sw = np.load('output/Swav_fvect.npy')
imgs_Sw = np.load('output/Swav_imgvect.npy')

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
#it would be good to add land cover type
#higher functional classification 
	# canids, felids, small herbivore, large herbivore, domestic animal

##split data by camera trap site
print('Organising data by site')
vecsbysite = {}
for site in sorted(set(ct_site)):
	vecsbysite[site] = features[ct_site==site]

specsbysite = {}
for site in sorted(set(ct_site)):
	specsbysite[site] = species[ct_site==site]

vecsbymgmt = {}
for site in sorted(set(mgmt)):
	vecsbymgmt[site] = features[mgmt==site]

specsbymgmt = {}
for site in sorted(set(mgmt)):
	specsbymgmt[site] = species[mgmt==site]

'''#save these
json.dumps(vecsbysite)
json.dumps(specsbysite)
json.dumps(vecsbymgmt)
json.dumps(specsbymgmt)
'''	

#create dataset with only day time images in.
#want hour to be >6 and <18
dayfeatures = []
dayhour = []
dayspecies = []
daymgmt = []
daysite = []
for i, v in enumerate(features):
	if time_hour[i] >= 6 and time_hour[i] <= 18:
		dayfeatures.append(v)
		dayhour.append(time_hour[i])
		dayspecies.append(species[i])
		daymgmt.append(mgmt[i])
		daysite.append(ct_site[i])


dayfeatures  = np.vstack(dayfeatures) #turn back into array


#create dataset with only wild animals

#create numpy array for important variables
ct_site_w = np.array(meta_wild.ct_site)
species_w = np.array(meta_wild.species)
mgmt_w = np.array(meta_wild.conservancy_name) 
time_hour_w = np.array(meta_wild.time_hour)
#it would be good to add land cover type
#higher functional classification 
	# canids, felids, small herbivore, large herbivore, domestic animal

#set empty lists
dayfeatures_w = []
dayhour_w = []
dayspecies_w = []
daymgmt_w = []
daysite_w = []

features_w = []
for i, v in enumerate(features):
	if species[i] not in domes:
		features_w.append(v)
#convert back to numpy
features_w  = np.vstack(features_w) #turn back into array

for i, v in enumerate(features_w):
	if time_hour_w[i] >= 6 and time_hour_w[i] <= 18:
		dayfeatures_w.append(v)
		dayhour_w.append(time_hour_w[i])
		dayspecies_w.append(species_w[i])
		daymgmt_w.append(mgmt_w[i])
		daysite_w.append(ct_site_w[i])


dayfeatures_w  = np.vstack(dayfeatures_w) #turn back into array


#####----Visualising entire dataset-----######
#create umap object
print('Plotting UMAP embeddings for entire dataset')
#we can plot using UMAP - this is not that good when you have lots of labels
u = umap.UMAP(random_state = 42).fit(features)
umap.plot.points(u)
u = umap.UMAP(n_neighbors = 30, min_dist = 0.0).fit(features)
umap.plot.points(u)

#plot - all images coloured by species
#note - fit transform here converts the output to a normal df, rather than UMAP object.
fit = umap.UMAP(random_state = 42)
u = fit.fit_transform(features) #this line can take a while

f, ax = plt.subplots(figsize=(10, 8))
sns.set_style("dark")
gpalette = sns.color_palette(cc.glasbey_bw, n_colors=32)
g = sns.scatterplot(x=u[:,0], y= u[:,1], hue=species, alpha=.6, palette=gpalette, s = 3, ax = ax)
sns.move_legend(g, "upper left", bbox_to_anchor=(1,1), frameon = False, fontsize = 10)
plt.title('PegNet Embeddings, coloured by species', fontsize = 12)
plt.tight_layout()
plt.savefig('output/figs/allimgs/PegNet/umap_species_PN.png', dpi='figure')


#sns.relplot(x=u[:,0], y= u[:,1], hue=species, alpha=.2, palette="muted", size = 1, ax = ax).set(title = 'PegNet embeddings coloured by species')

#plot - all coloured by ct_site
#skipping because it doesn't show anything meaningful - too many data points
'''
sns.set_theme(style="white")\
hue_order = sorted(ct_site)
sns.relplot(x=u[:,0], y= u[:,1], hue=ct_site, hue_order=hue_order, alpha=.2, palette="mako",
            height=10).set(title = 'UMAP embedding coloured by site')
plt.savefig('output/figs/allimgs/umap_sites_Emb.png', dpi='figure')
'''

#plots coloured by management zone
f, ax = plt.subplots(figsize=(10, 8))
sns.set_style("dark")
g = sns.scatterplot(x=u[:,0], y= u[:,1], hue=mgmt, hue_order = ['NP', 'BZ','OBZ'], alpha=.6, palette='viridis', s = 3, ax = ax)
sns.move_legend(g, "upper left", bbox_to_anchor=(1,1), frameon = False, fontsize = 10)
plt.title('PegNet Embeddings, coloured by Management Regime', fontsize = 12)
plt.tight_layout()
plt.savefig('output/figs/allimgs/PegNet/umap_mgmt_PN.png', dpi='figure')

#sns.relplot(x=u[:,0], y= u[:,1], hue=mgmt, hue_order = hue_order, alpha=.2, palette="mako",
            height=10).set(title = 'UMAP embedding coloured by management zone')


#plots coloured by time of day
#custom colour plot 
color_list = ['#000000', '#380000', '#560000', '#760100', '#980300', '#bb0600', '#df0d00', '#f93500', '#fe6800', '#ff9100', '#ffb402', '#ffd407', '#ffd407', '#ffb402', '#ff9100','#fe6800', '#f93500', '#df0d00','#bb0600','#980300','#760100', '#560000', '#380000','#000000']

f, ax = plt.subplots(figsize=(10, 8))
sns.set_style("dark")
gpalette = sns.color_palette(color_list)
g = sns.scatterplot(x=u[:,0], y= u[:,1], hue=time_hour, palette = gpalette, alpha=.6, s = 3, ax = ax)
sns.move_legend(g, "upper left", bbox_to_anchor=(1,1), frameon = False, fontsize = 10)
plt.title('PegNet Embeddings, coloured by time of day (hour)', fontsize = 12)
plt.tight_layout()
plt.savefig('output/figs/allimgs/PegNet/umap_hour_PN.png', dpi='figure')

print('Plots (entire dataset) saved!')

#####-----Plotting features from day time images only----######
features = dayfeatures
#create umap object
print('Plotting UMAP embeddings for daytime images')
#we can plot using UMAP - this is not that good when you have lots of labels
u = umap.UMAP(random_state = 42).fit(features)
umap.plot.points(u)
u = umap.UMAP(n_neighbors = 30, min_dist = 0.0).fit(features)
umap.plot.points(u)

#note - fit transform here converts the output to a normal df, rather than UMAP object.
fit = umap.UMAP(random_state = 42, min_dist = 0.0)
u = fit.fit_transform(features) #this line can take a while

#plot - all images coloured by species
#create order the same as other plot
hue_order = ['chital','indian_grey_mongoose','macaque','goat','bird', 'cow',
'wild_boar', 'buffalo', 'peacock', 'dog', 'domestic_cat', 'jackal',
'jungle_cat', 'jungle_fowl', 'sheep', 'domestic_chicken', 'hare', 'grey_langur',
'nilgai','sambar', 'barking_deer',  'four_horned_antelope',  'leopard',  
'one_horned_rhino','elephant', 'tiger', 'hog_deer', 'porcupine', 'sloth_bear']
#note - this makes nearly the same colours as full dataset plot, 
#but need to find a way to add gaps


f, ax = plt.subplots(figsize=(10, 8))
sns.set_style("dark")
gpalette = sns.color_palette(cc.glasbey_bw, n_colors=29)
g = sns.scatterplot(x=u[:,0], y= u[:,1], hue=dayspecies, hue_order = hue_order, alpha=.6, palette=gpalette, s = 3, ax = ax)
sns.move_legend(g, "upper left", bbox_to_anchor=(1,1), frameon = False, fontsize = 10)
plt.title('PegNet Embeddings, day time only, coloured by species', fontsize = 12)
plt.tight_layout()
plt.savefig('output/figs/allimgs/PegNet/umap_species_PNd.png', dpi='figure')

#plots coloured by management zone
f, ax = plt.subplots(figsize=(10, 8))
sns.set_style("dark")
g = sns.scatterplot(x=u[:,0], y= u[:,1], hue=daymgmt, hue_order = ['NP', 'BZ','OBZ'], alpha=.6, palette='viridis', s = 3, ax = ax)
sns.move_legend(g, "upper left", bbox_to_anchor=(1,1), frameon = False, fontsize = 10)
plt.title('PegNet Embeddings, day time only, coloured by Management Regime', fontsize = 12)
plt.tight_layout()
plt.savefig('output/figs/allimgs/PegNet/umap_mgmt_PNd.png', dpi='figure')

#sns.relplot(x=u[:,0], y= u[:,1], hue=mgmt, hue_order = hue_order, alpha=.2, palette="mako",
            height=10).set(title = 'UMAP embedding coloured by management zone')

print('PegNet day time plots saved!')


#######----Plotting features from wild animals only---#####
print('Plotting UMAP embeddings for entire dataset')
fit = umap.UMAP(min_dist = 0.0, random_state = 42)
u = fit.fit_transform(features_w) #this line can take a while
plot = sns.relplot(x = u[:,0], y = u[:,1]).set(title = 'UMAP embedding')
plot = sns.relplot(x = u[:,0], y = u[:,1], hue = species_w, alpha=0.2).set(title = 'UMAP embedding')

features = features_w
#create umap object
print('Plotting UMAP embeddings for wild species only')
#we can plot using UMAP - this is not that good when you have lots of labels

#note - fit transform here converts the output to a normal df, rather than UMAP object.
fit = umap.UMAP(random_state = 42, min_dist = 0.0)
u = fit.fit_transform(features) #this line can take a while

#plot - all images coloured by species
#create order the same as other plot
hue_order = ['chital','indian_grey_mongoose','macaque','goat','bird', 'cow',
'wild_boar', 'buffalo', 'peacock', 'dog', 'domestic_cat', 'jackal',
'jungle_cat', 'jungle_fowl', 'sheep', 'domestic_chicken', 'hare', 'grey_langur',
'nilgai','sambar', 'barking_deer',  'four_horned_antelope',  'leopard',  
'one_horned_rhino','elephant', 'tiger', 'hog_deer', 'porcupine', 'sloth_bear']
#note - this makes nearly the same colours as full dataset plot, 
#but need to find a way to add gaps


f, ax = plt.subplots(figsize=(10, 8))
sns.set_style("dark")
gpalette = sns.color_palette(cc.glasbey_bw, n_colors=25)
g = sns.scatterplot(x=u[:,0], y= u[:,1], hue=species_w, alpha=.6, palette=gpalette, s = 3, ax = ax)
sns.move_legend(g, "upper left", bbox_to_anchor=(1,1), frameon = False, fontsize = 10)
plt.title('PegNet Embeddings, wild species only, coloured by species', fontsize = 12)
plt.tight_layout()
plt.savefig('output/figs/allimgs/PegNet/umap_species_PNw.png', dpi='figure')

#plots coloured by management zone
f, ax = plt.subplots(figsize=(10, 8))
sns.set_style("dark")
g = sns.scatterplot(x=u[:,0], y= u[:,1], hue=mgmt_w, hue_order = ['NP', 'BZ','OBZ'], alpha=.6, palette='viridis', s = 3, ax = ax)
sns.move_legend(g, "upper left", bbox_to_anchor=(1,1), frameon = False, fontsize = 10)
plt.title('PegNet Embeddings, wild species only, coloured by Management Regime', fontsize = 12)
plt.tight_layout()
plt.savefig('output/figs/allimgs/PegNet/umap_mgmt_PNw.png', dpi='figure')

#sns.relplot(x=u[:,0], y= u[:,1], hue=mgmt, hue_order = hue_order, alpha=.2, palette="mako",
            height=10).set(title = 'UMAP embedding coloured by management zone')

print('PegNet day time plots saved!')


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



#testing stability of k predictions:
#everytime you reduce features using UMAP, the output is based on a random sequence
#however, hopefully the points which are similar to each other will come out the same.
#if optimal k comes out the same each time, then the analysis is stable.

for i in range(1,5):
	x = vecsbysite['BZ03']
	kmax = 30
	dim = 512 #need to reduce dimensions
	embedding = umap.UMAP(n_components=dim).fit_transform(x)
	for k in range(2, kmax+1):
  		kmeans = cluster.KMeans(n_clusters = k).fit(embedding)
  		labels = kmeans.labels_ #underscore is needed
  		sil.append(metrics.silhouette_score(embedding, labels, metric = 'euclidean'))
	#find index of max silhouette and add 2 (0 = 2)
	optimk = sil.index(max(sil))+2
	print(f'{i} = {optimk}')
'''
#output:
1 = 3
2 = 3
3 = 3
4 = 3
5 = 3
6 = 3
7 = 3
'''

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



#possible parameters
#clusterable_embedding = umap.UMAP(
    #n_neighbors=30,
    #min_dist=0.0,
    #n_components=4,
    #random_state=42,
#).fit_transform(x)

#HBDSCAN clustering




#repeat for Swav and for iNat model





