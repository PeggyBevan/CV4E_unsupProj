#testing running a script in VM

import os

#set wd
load_dir = '/home/pegbev/nepal19_crops'

#list files
image_list = os.listdir(load_dir)
#print length of all unfiltered images
print('Number of images (unfiltered):', len(image_list))

#doing the same thing, but only files ending in .jpg, to make sure there are no hidden files
image_list = [name for name in image_list if '.jpg' in name] # list comprehension
print(len(image_list))

#create a new list of images, and only choose those which don't contain human or vehicle
image_list_filtered = []
for name in image_list:
	if not '_human' in name and not '_vehicle' in name:
		image_list_filtered.append(name)
	
print('Number of images (filtered):', len(image_list_filtered))