'''
	Taking in feature vectors from model and visualising using UMAP
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import umap
%matplotlib inline

#read in .pkl file
data = ...pk
#create umap object
fit = umap.UMAP()
%time u = fit.fit_transform(data)

#initial plot
plt.scatter(u[:,0], u[:,1], c=data)
plt.title('UMAP embedding CT images');