##3/08/2022
# Labelling data as train, val, test. 
# The dataframe has a column called SetID which has three levels: 'test', 'val' and 'train'
# I want to create a text file.

#Packages
#import numpy as np
import pandas as pd
import os

#Load csv - pandas
#VMdir = '/home/pegbev/data/'
locdir = '/Users/peggybevan/Documents/Github/CV4E_unsupProj/data/'
excel = 'nepal_cropsmeta_PB.csv'
data = pd.read_csv(locdir + excel)
print('Total length:',len(data))


#remove paths containging human or vehicle.
anthro = ['human', 'vehicle']	
domes = ['buffalo', 'cow', 'dog', 'domestic_cat', 'domestic_chicken', 'domestic elephant', 'goat', 'sheep']

df = data[-data['species'].isin(anthro)]
dfwild = df[-df['species'].isin(domes)]
print('Total length:',len(df))
print('Total length (wild only):',len(dfwild))
	
#update img_path to fit with hard drive

#create csv with img_path, species, ct_site
train = df[df.SetID == 'train']
train = train[['img_path', 'species', 'ct_site']]
print('Train csv length:', len(train))


train_wild = dfwild[dfwild.SetID == 'train']
train_wild = train_wild[['img_path', 'species', 'ct_site']]
print('Train csv length (wild only):', len(train_wild))

#save
path = "data"
# Check whether the specified path exists or not
if not os.path.exists(path):
   # Create a new directory because it does not exist
   os.makedirs(path)

train.to_csv('data/train.csv')
train_wild.to_csv('data/train_wild.csv')

#create a list of image paths for the model to read
#If SetID contains 'train', list img_path
train_list = []
for index, row in df.iterrows():
	if 'train' in row['SetID']:
		train_list.append(row['img_path'] + ','  + '\n')
		train_list.append

print('Total train set size:', len(train_list))

trainwild_list = []
for index, row in dfwild.iterrows():
	if 'train' in row['SetID']:
		trainwild_list.append(row['img_path'] + ','  + '\n')
		trainwild_list.append

print('Total train set size (wild only):', len(trainwild_list))


#save as text file 
f = open('data/train.txt', 'a')
f.writelines(train_list)
f.close()

#save as text file 
fw = open('data/train_wild.txt', 'a')
fw.writelines(trainwild_list)
fw.close()

#create csv with img_path, species, ct_site
val = df[df.SetID == 'val']
val = val[['img_path', 'species', 'ct_site']]
print('Val csv length', len(val))

#save
val.to_csv('data/val.csv')

#If SetID contains 'val', list img_path
val_list = []
for index, row in df.iterrows():
	if 'val' in row['SetID']:
		val_list.append(row['img_path'] + '\n')

print('Total val set size:', len(val_list))

#save as text file 
f = open('data/val.txt', 'a')
f.writelines(val_list)
f.close()

#create csv with img_path, species, ct_site
valwild = dfwild[dfwild.SetID == 'val']
valwild = valwild[['img_path', 'species', 'ct_site']]
print('Val csv length (wild only)', len(valwild))

#save
valwild.to_csv('data/val_wild.csv')

#If SetID contains 'val', list img_path
valwild_list = []
for index, row in dfwild.iterrows():
	if 'val' in row['SetID']:
		valwild_list.append(row['img_path'] + '\n')

print('Total val set size (wild only):', len(valwild_list))

#save as text file 
f = open('data/val_wild.txt', 'a')
f.writelines(valwild_list)
f.close()

#create csv with img_path, species, ct_site
test = df[df.SetID == 'test']
test = test[['img_path', 'species', 'ct_site']]
print('Test csv length', len(test))

#save
test.to_csv('data/test.csv')

#If SetID contains 'test', list img_path
test_list = []
for index, row in df.iterrows():
	if 'test' in row['SetID']:
		test_list.append(row['img_path'] + '\n')

print('Total test set size:', len(test_list))

#save as text file 
f = open('data/test.txt', 'a')
f.writelines(test_list)
f.close()


#create csv with img_path, species, ct_site
testwild = dfwild[dfwild.SetID == 'test']
testwild = testwild[['img_path', 'species', 'ct_site']]
print('Test csv length (wild only)', len(testwild))

#save
testwild.to_csv('data/test_wild.csv')

#If SetID contains 'test', list img_path
testwild_list = []
for index, row in dfwild.iterrows():
	if 'test' in row['SetID']:
		testwild_list.append(row['img_path'] + '\n')

print('Total test set size (wild only):', len(testwild_list))

#save as text file 
f = open('data/test_wild.txt', 'a')
f.writelines(testwild_list)
f.close()