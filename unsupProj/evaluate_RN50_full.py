'''
	Taking in feature vectors from model and visualising using UMAP - PegNet and Embnet
	Peggy Bevan Aug 2022
'''
#if numpy files need importing, in cmd line:
#scp -i ~/Downloads/PegNetVM_key.pem "pegbev@20.228.65.187:/home/pegbev/output/PegNet_fvect_norm.npy" "/Users/peggybevan/OneDrive/My Documents/PhD/CameraTraps/CV4E/output/"
#scp -i ~/Downloads/PegNetVM_key.pem "pegbev@20.228.65.187:/home/pegbev/output/PegNet_imgvect_norm.npy" "/Users/peggybevan/OneDrive/My Documents/PhD/CameraTraps/CV4E/output/"
#scp -i ~/Downloads/PegNetVM_key.pem "pegbev@20.228.65.187:/home/pegbev/output/Swav_fvect.npy" "/Users/peggybevan/OneDrive/My Documents/PhD/CameraTraps/CV4E/output/"
#scp -i ~/Downloads/PegNetVM_key.pem "pegbev@20.228.65.187:/home/pegbev/output/Swav_imgvect.npy" "/Users/peggybevan/OneDrive/My Documents/PhD/CameraTraps/CV4E/output/"

#%% [markdown]
# ## Import packages and data
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
import json
import hdbscan
import sklearn.cluster as cluster
from collections import defaultdict
from sklearn import metrics, datasets
from sklearn.metrics import pairwise_distances, adjusted_rand_score, adjusted_mutual_info_score
import colorcet as cc

#%matplotlib auto #display plots

#%% 
#read in numpy arrays - output from predict.py
print('reading in feature vectors')

features = np.load('output/NP_RN50_full_fvect.npy')
imgs = np.load('output/NP_RN50_full_imgvect.npy')

#add information about camera trap location and species
meta = pd.read_csv('data/nepal_cropsmeta_PB.csv')
meta.head()
#%%
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
#%%
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
#%%

#####----Visualising entire dataset - PegNet-----######
#create umap object
print('Plotting UMAP embeddings for entire dataset - RN50 supervised')
#we can plot using UMAP - this is not that good when you have lots of labels
#u = umap.UMAP(random_state = 42).fit(features)
#umap.plot.points(u)
#u = umap.UMAP(n_neighbors = 30, min_dist = 0.0).fit(features)
#umap.plot.points(u)

#plot - all images coloured by species
#note - fit transform here converts the output to a normal df, rather than UMAP object.
fit = umap.UMAP(random_state = 42)
u = fit.fit_transform(features) #this line can take a while
#%%
f, ax = plt.subplots(figsize=(10, 8))
sns.set_style("white")
gpalette = sns.color_palette(cc.glasbey_bw, n_colors=32)
g = sns.scatterplot(x=u[:,0], y= u[:,1], hue=species, alpha=.6, palette=gpalette, s = 3, ax = ax)
plt.legend(markerscale=3)
sns.move_legend(g, "upper left", bbox_to_anchor=(1,1), frameon = False, fontsize = 12)
plt.title('ResNet50 supervised Embeddings, coloured by species', fontsize = 12)

plt.tight_layout()
plt.savefig('output/figs/RN50_full/umap_species_RN50.png', dpi='figure')

#%%
#create colour palette
# Unique category labels: 'D', 'F', 'G', ...
color_labels = meta['species'].unique()
# List of RGB triplets
rgb_values = sns.color_palette("Set2", 32)
# Map label to RGB
color_map = dict(zip(color_labels, rgb_values))
speciesPD = pd.DataFrame(species, columns = ['species'])

def draw_umap(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', title=''):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    u = fit.fit_transform(features);
    fig = plt.figure()
    if n_components == 1:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], range(len(u)), c=speciesPD['species'].map(color_map), s = 2)
    if n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], u[:,1], c=speciesPD['species'].map(color_map), s = 2, alpha = 0.2)
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(u[:,0], u[:,1], u[:,2], c=speciesPD['species'].map(color_map), s=2, alpha = 0.2)
    plt.title(title, fontsize=18)
#%%
	
for n in (10, 20, 50, 100, 200):
    draw_umap(n_neighbors=n, title='n_neighbors = {}'.format(n))

#%%
for n in (0.0, 0.1, 0.5, 0.7, 0.99):
    draw_umap(min_dist=n, title='min_dist= {}'.format(n))
#low min_dist = more compact clusters	
#%%
#playing around with these parameters doesn't seem to make an enormous difference - but n_neighbours = 50 or N-neighbours = 10 and min_dist = 0.1 seems to be the best	
draw_umap(min_dist= 0.1, n_neighbors = 50, title='n_neighbors = 50, min_dist = 0.1')	
draw_umap(min_dist= 0.1, n_neighbors = 10, title='n_neighbors = 10, min_dist = 0.1')
#using n_neighbours = 50 and min_dist = 0.1 gives more separation of clusters within the main body. 	
#%%
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
plt.title('ResNet50 supervised Embeddings, coloured by Management Regime', fontsize = 12)
plt.tight_layout()
plt.savefig('output/figs/RN50_full/umap_mgmt_RN50.png', dpi='figure')

#sns.relplot(x=u[:,0], y= u[:,1], hue=mgmt, hue_order = hue_order, alpha=.2, palette="mako", height=10).set(title = 'UMAP embedding coloured by management zone')


#plots coloured by time of day
#custom colour plot 
color_list = ['#000000', '#380000', '#560000', '#760100', '#980300', '#bb0600', '#df0d00', '#f93500', '#fe6800', '#ff9100', '#ffb402', '#ffd407', '#ffd407', '#ffb402', '#ff9100','#fe6800', '#f93500', '#df0d00','#bb0600','#980300','#760100', '#560000', '#380000','#000000']

f, ax = plt.subplots(figsize=(10, 8))
sns.set_style("dark")
gpalette = sns.color_palette(color_list)
g = sns.scatterplot(x=u[:,0], y= u[:,1], hue=time_hour, palette = gpalette, alpha=.6, s = 3, ax = ax)
sns.move_legend(g, "upper left", bbox_to_anchor=(1,1), frameon = False, fontsize = 10)
plt.title('ResNet50 supervised Embeddings, coloured by time of day (hour)', fontsize = 12)
plt.tight_layout()
plt.savefig('output/figs/RN50_full/umap_hour_RN.png', dpi='figure')

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
plt.title('ResNet50 supervised Embeddings, day time only, coloured by species', fontsize = 12)
plt.tight_layout()
plt.savefig('output/figs/RN50_full/umap_species_RNd.png', dpi='figure')

#plots coloured by management zone
f, ax = plt.subplots(figsize=(10, 8))
sns.set_style("dark")
g = sns.scatterplot(x=u[:,0], y= u[:,1], hue=daymgmt, hue_order = ['NP', 'BZ','OBZ'], alpha=.6, palette='viridis', s = 3, ax = ax)
sns.move_legend(g, "upper left", bbox_to_anchor=(1,1), frameon = False, fontsize = 10)
plt.title('ResNet50 supervised Embeddings, day time only, coloured by Management Regime', fontsize = 12)
plt.tight_layout()
plt.savefig('output/figs/RN50_full/umap_mgmt_RNd.png', dpi='figure')

#sns.relplot(x=u[:,0], y= u[:,1], hue=mgmt, hue_order = hue_order, alpha=.2, palette="mako", height=10).set(title = 'UMAP embedding coloured by management zone')

print('ResNet day time plots saved!')


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
plt.title('ResNet50 supervised  Embeddings, wild species only, coloured by species', fontsize = 12)
plt.tight_layout()
plt.savefig('output/figs/allimgs/PegNet/umap_species_PNw.png', dpi='figure')

#plots coloured by management zone
f, ax = plt.subplots(figsize=(10, 8))
sns.set_style("dark")
g = sns.scatterplot(x=u[:,0], y= u[:,1], hue=mgmt_w, hue_order = ['NP', 'BZ','OBZ'], alpha=.6, palette='viridis', s = 3, ax = ax)
sns.move_legend(g, "upper left", bbox_to_anchor=(1,1), frameon = False, fontsize = 10)
plt.title('ResNet50 supervised  Embeddings, wild species only, coloured by Management Regime', fontsize = 12)
plt.tight_layout()
plt.savefig('output/figs/allimgs/PegNet/umap_mgmt_PNw.png', dpi='figure')

print('ResNet50 supervised  wild plots saved!')
# %%
