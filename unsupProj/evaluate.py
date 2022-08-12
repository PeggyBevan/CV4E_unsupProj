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



#read in .pkl file
data = []
file_name = "../output/main_output.pt"

with (open(file_name, "rb")) as f:
    data = pickle.load(f)

#add information about camera trap location and species
#meta =  
excel = 'data/nepal_cropsmeta_PB.csv'
meta = pd.read_csv(excel)


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
nmfeats = data[0]['features'].numpy()
%time u = fit.fit_transform(nmfeats)

#initial plot
plot = plt.scatter(u[:,0], u[:,1], c=nmfeats)
plt.title('UMAP embedding CT images');


for batch in data.keys():
	data[batch]['ct_site'] = []
	data[batch]['species'] = []
	for file in data[batch]['img_path']:
		x = meta.loc[meta['img_path']==file]['ct_site'].to_string(index = False)
		data[batch]['ct_site'].append(x)
		y = meta.loc[meta['img_path']==file]['species'].to_string(index = False)
		data[batch]['species'].append(y)


mapping species names to values
spec_val = defaultdict(lambda: len(spec_val))
spec_val = [spec_val[ele] for ele in sorted(set(meta.species))]
zip_specvals = zip(spec_val, sorted(set(meta.species)
zip_specvals = list(zip_specvals)


new_dict = {}
new_dict['features'] = []
new_dict['img_path'] = []
new_dict['ct_site'] = []
new_dict['species'] = []

site_tofeatures = {}
site_toSpecies = {}
for batch in data.keys():
	for features, img_path, ct_site, species in zip(data['features'], data['img_path'], data['ct_site'], data['species']):
		site_tofeatures[ct_site].append(features) 	
		site_tofeatures[ct_site].append(species)

for  batch in data.keys():
	cat or stack 
	zip for splitting data



#to convert factors to unique numbers
from collections import defaultdict
site_val = defaultdict(lambda: len(site_val))
site_val = [site_val[ele] for ele in data[0]['ct_site']]

plot = plt.scatter(u[:,0], u[:,1], c=site_val)

spec_val = defaultdict(lambda: len(spec_val))
spec_val = [spec_val[ele] for ele in data[0]['species']]

plot = plt.scatter(u[:,0], u[:,1], c=spec_val)




