##3/08/2022
# Labelling data as train, val, test. 
# The dataframe has a column called SetID which has three levels: 'test', 'val' and 'train'
# I want to create a text file.

#Packages
#import numpy as np
import pandas as pd
#import os

#Load csv - pandas
VMdir = '/home/pegbev/data/'
locdir = '/Users/peggybevan/OneDrive/My Documents/PhD/CameraTraps/CV4E/data/'
excel = 'nepal_cropsmeta_PB.csv'
data = pd.read_csv(VMdir + excel)

print('Total length:',len(data))

#remove paths containging human or vehicle.
anthro = ['human', 'vehicle']
	
dfwild = data[-data['species'].isin(anthro)]

print('Total length (wild):',len(dfwild))
	
#If SetID contains 'train', list img_path
train_list = []
for index, row in dfwild.iterrows():
	if 'train' in row['SetID']:
		train_list.append(row['img_path'] + '\n')

print('Total train set size:', len(train_list))

#save as text file 
f = open('/home/pegbev/data/train.txt', 'a')
f.writelines(train_list)
f.close()

#If SetID contains 'val', list img_path
val_list = []
for index, row in dfwild.iterrows():
	if 'val' in row['SetID']:
		val_list.append(row['img_path'] + '\n')

print('Total val set size:', len(val_list))

#save as text file 
f = open('/home/pegbev/data/val.txt', 'a')
f.writelines(val_list)
f.close()


#If SetID contains 'test', list img_path
test_list = []
for index, row in dfwild.iterrows():
	if 'test' in row['SetID']:
		test_list.append(row['img_path'] + '\n')

print('Total test set size:', len(test_list))

#save as text file 
f = open('/home/pegbev/data/test.txt', 'a')
f.writelines(test_list)
f.close()


