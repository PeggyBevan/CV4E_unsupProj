'''
	Taking in feature vectors from model and visualising using UMAP
'''

import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
!pip install umap-learn
import umap
%matplotlib inline


#read in numpy arrays
features = np.load('output/featurevectors.npy')
imgs = np.load('output/imgpathvectors.npy')

'''
#read in .pkl file
data = []
file_name = "output/main_output_dict.pt"

with (open(file_name, "rb")) as f:
    data = pickle.load(f)
'''

#add information about camera trap location and species
excel = 'data/nepal_cropsmeta_PB.csv'
meta = pd.read_csv(excel)

#meta has humans and vehicles in - remove
anthro = ['human', 'vehicle']
meta = meta[-meta['species'].isin(anthro)]
#only training
meta = meta[meta['SetID']=='train']

#takeout variables
ct_site = np.array(meta.ct_site)
species = np.array(meta.species)
mgmt = np.array(meta.conservancyname) 


for batch in data.keys():
	data[batch]['ct_site'] = []
	data[batch]['species'] = []
	for file in data[batch]['img_path']:
		x = meta.loc[meta['img_path']==file]['ct_site']
		data[batch]['ct_site'].append(x) 
		y = meta.loc[meta['img_path']==file]['species']
		data[batch]['species'].append(y)
	

#create umap object
fit = umap.UMAP()
%time u = fit.fit_transform(nmfeats)

#initial plot
plot = plt.scatter(u[:,0], u[:,1], c=nmfeats)
plt.title('UMAP embedding CT images');

'''
for batch in data.keys():
	data[batch]['ct_site'] = []
	data[batch]['species'] = []
	for file in data[batch]['img_path']:
		x = meta.loc[meta['img_path']==file]['ct_site'].to_string(index = False)
		data[batch]['ct_site'].append(x)
		y = meta.loc[meta['img_path']==file]['species'].to_string(index = False)
		data[batch]['species'].append(y)
'''

#mapping species names to values
from collections import defaultdict
spec_val = defaultdict(lambda: len(spec_val))
spec_val = [spec_val[ele] for ele in sorted(set(meta.species))]
zip_specvals = zip(sorted(set(meta.species)), spec_val)
zip_specvals = dict(zip_specvals)

species1 = []
for i in range(len(species)):
	a = zip_specvals[species1]
	species1.append(a)

#plot - all images
plt.scatter(u[:,0], u[:,1], c=species1, alpha = 0.1)

##split 

#to convert factors to unique numbers
from collections import defaultdict
site_val = defaultdict(lambda: len(site_val))
site_val = [site_val[ele] for ele in data[0]['ct_site']]

plot = plt.scatter(u[:,0], u[:,1], c=site_val)

spec_val = defaultdict(lambda: len(spec_val))
spec_val = [spec_val[ele] for ele in data[0]['species']]

plot = plt.scatter(u[:,0], u[:,1], c=spec_val)




