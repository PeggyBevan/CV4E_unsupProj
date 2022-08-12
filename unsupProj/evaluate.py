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
file_name = "output/test_output.p"

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

newlist = []
for batch in data.keys():
	data[batch]['ct_site'] = []
	data[batch]['species'] = []
	for file in data[batch]['img_path']:
		x = meta.loc[meta['img_path']==file]['ct_site']
		data[batch]['ct_site'].append(x) 
		y = meta.loc[meta['img_path']==file]['species']
		data[batch]['species'].append(y)