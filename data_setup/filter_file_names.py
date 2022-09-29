##THIS is a file of notes on different types of code

#Python
##test file - make list of file names without anthro catergories and list them
import os

load_dir = '/Volumes/File_backups/Peggy/PhD/CV4E/CroppedImgs/nepal_crops_and_df/nepal19_crops'

image_list = os.listdir(load_dir)
print(len(image_list))
image_list = [name for name in image_list if '.jpg' in name] # list comprehension
print(len(image_list))

image_list_filtered = []
for name in image_list:
	if not '_human' in name and not '_vehicle' in name:
		image_list_filtered.append(name)
	
print(len(image_list_filtered))

#list unique values 
(set(data.column))
#number of objects 
len(set(data.column))

#give column names panda
df.columns

#mapping values to integers for mat plot plottinghy1		
spec_val = defaultdict(lambda: len(spec_val))
spec_val = [spec_val[ele] for ele in sorted(set(meta.species))]
zip_specvals = zip(sorted(set(meta.species)), spec_val)
zip_specvals = dict(zip_specvals)

species1 = []
for i in range(len(species)):
	a = zip_specvals[species[i]]
	species1.append(a)

#mapping ct_site names to values
site_val = defaultdict(lambda: len(site_val))
site_val = [site_val[ele] for ele in sorted(set(meta.ct_site))]
zip_sitevals = zip(sorted(set(meta.ct_site)), site_val)
zip_sitevals = dict(zip_sitevals)

site_val1 = []
for i in range(len(ct_site)):
	a = zip_sitevals[ct_site[i]]
	site_val1.append(a)

##COMMAND LINE CODE

#log in to VM
ssh -i Downloads/PegNetVM_key.pem pegbev@20.228.65.187

#to upload entire directory azure
az storage blob directory upload --account-name peggybevanblob 
--account-key wlVsMeCUJV/6/3E/9ldzZB6XCsq8yHpGu+retHc34gMt/9Il9w1MeyxYbpIIEP5tTqjZGJLjEOaL+AStypEuiQ== 
--container peggybevanblob --destination-path nepal19_crops 
--source /Volumes/File_backups/Peggy/PhD/CV4E/CroppedImgs/nepal_crops_and_df/nepal19_crops

#upload entire directory to VM 
azcopy cp "https://peggybevanblob.blob.core.windows.net/peggybevanblob/nepal19_crops?sv=2021-06-08&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2023-08-03T06:48:13Z&st=2022-08-02T22:48:13Z&spr=https&sig=vVoYwWS6NZmZb63H9gY%2BBUSBpAyyxiKq7RWZDjLuv84%3D" "" --recursive=true

##scp - copy files directly to virtual machine, rather than via blob. 
scp -i Downloads/PeggyBevanDSImg01_key.pem .ssh/id_ed25519.pub pegbev@20.29.185.98:/home/pegbev/.ssh/

#copying full dataset                                
scp -i Downloads/PeggyBevanDSImg01_key.pem /Volumes/File_backups/Peggy/PhD/CV4E/CroppedImgs/nepal_crops_and_df/nepal_filtered_meta.csv	pegbev@20.29.185.98:/home/pegbev/nepal_filtered_meta.csv

#copying full dataset with train/test setID
scp -i Downloads/PeggyBevanDSImg01_key.pem '/Users/peggybevan/OneDrive/My Documents/PhD/CameraTraps/CV4E/data/nepal_cropsmeta_PB.csv' pegbev@20.29.185.98:/home/pegbev/data/nepal_cropsmeta_PB.csv

#moving folder up one level
mv nepal19_crops/nepal19_crops nepal19_crops/crops
mv nepal19_crops/crops /home/pegbev/crops
rmdir nepal19_crops
mv crops nepal19_crops

#copying from VM to local
scp -i ~/Downloads/PegNetVM_key.pem "pegbev@20.228.65.187:/home/pegbev/output/main_output_dict.pt" "/Users/peggybevan/OneDrive/My Documents/PhD/CameraTraps/CV4E/output/"
scp -i ~/Downloads/PegNetVM_key.pem "pegbev@20.228.65.187:/home/pegbev/output/kmeans_nspecies_comp.csv" "/Users/peggybevan/OneDrive/My Documents/PhD/CameraTraps/CV4E/output/"
scp -i Downloads/PegNetVM_key.pem Downloads/ssl_camera_trap_weights/context_positives/kenya_resnet50_simclr_2022_05_05__16_34_13.pt pegbev@220.228.65.187:/home/pegbev/data/kenya_resnet50_simclr_2022_05_05__16_34_13.pt
scp -i ~/Downloads/PegNetVM_key.pem "pegbev@20.228.65.187:/home/pegbev/output/kmeans_EB_dims.csv" "/Users/peggybevan/OneDrive/My Documents/PhD/CameraTraps/CV4E/output/"
scp -i ~/Downloads/PegNetVM_key.pem "pegbev@20.228.65.187:/home/pegbev/output/kmeans_PN_dims.csv" "/Users/peggybevan/OneDrive/My Documents/PhD/CameraTraps/CV4E/output/PegNet"
scp -i ~/Downloads/PegNetVM_key.pem "pegbev@20.228.65.187:/home/pegbev/output/Emb_fvect.npy" "/Users/peggybevan/OneDrive/My Documents/PhD/CameraTraps/CV4E/output/"
scp -i ~/Downloads/PegNetVM_key.pem "pegbev@20.228.65.187:/home/pegbev/output/Emb_imgvect.npy" "/Users/peggybevan/OneDrive/My Documents/PhD/CameraTraps/CV4E/output/PegNet"
scp -i ~/Downloads/PegNetVM_key.pem "pegbev@20.228.65.187:/home/pegbev/output/kmeans_PN_day.csv" "/Users/peggybevan/OneDrive/My Documents/PhD/CameraTraps/CV4E/output/PegNet"


## GIT HUB
#make sure to add repo by SSH rather than HTTPS 
git init 
git remote add CV4E git@github.com:PeggyBevan/CV4E_unsupProj.git

#when you add a new file to repository, it will be untracked
git add file.py 

git commit -m 'comment about commit'

#to upload online:
git push <nickname>





                                 