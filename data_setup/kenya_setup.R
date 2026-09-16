#getting kenya data set up.

kenya = read.csv('data/kenya/kenya_clf_train_crops/kenya_filtered_v9_sampled_w_other.csv')
kenya_noother = read.csv('data/kenya/kenya_clf_train_crops/kenya_filtered_v9_sampled_noother.csv')

nrow(kenya)
nrow(kenya_noother)
names(kenya)
crops_files = list.files('data/kenya/kenya_clf_train_crops/kenya18_crops_v9')
length(crops_files)
tags = read.csv('data/kenya/kenya_clf_train_crops/kenya18_crops_v9/tags_all.csv')
tags_animals = read.csv('data/kenya/kenya_clf_train_crops/kenya18_crops_v9/tags_only_animals.csv')
nrow(tags_animals)
crops_files[1]

# what is in nepal?
nepal_meta = read.csv('data/nepal_cropsmeta_PB.csv')
names(nepal_meta)
unique(nepal_meta$species)


# need a dataframe with the list of crops (tags_animals) joined with metadata for the image names (kenya)

kenya$img_id[1]
tags_animals$img_id[1]

library(dplyr)
tags_meta = tags_animals %>% left_join(kenya[,c(16,18,19:36)], by = 'box_name')
# check for any that didn't align
missing = tags_meta[is.na(tags_meta$location),]
miss2 = kenya[is.na(kenya$img_set),]
# so there are about 81000 images that are not in kenya
unique(missing$species)


# summary of dataframe
# number of camtrap sites
unique(tags_meta$location)
# 171
table(tags_meta$conservancy)
# MN    MT    NB   OMC 
# 20344 18323 27310 17479 
# even spread across conservancies

unique(tags_meta$img_set)
table(tags_meta$img_set)
# test train   val 
# 25045 52589  5822 
# there are 52k labels labelled as train
table(nepal_meta$SetID)
# test train   val 
# 15153 43394  8412 

# date range
range(tags_meta$date, na.rm = TRUE)
# october - november 2018 only.

# lets only keep the rows we need for the model training
tags_meta = tags_meta %>%
  filter(!is.na(img_name))

unique(tags_meta$species)
# there are 98 species in this dataset!
table(tags_meta$species)

tags_meta$box_path[1]

# so for every image in the train test we want a csv with img_path, species and ct_site
# edit box_path to this repo
tags_meta$box_path[1]
tags_meta$box_path_PB = gsub("/home/omi/projects/camera_traps_classifier/data/kenya/", "kenya/kenya_clf_train_crops/", tags_meta$box_path)
tags_meta$box_path_PB[1]

train = tags_meta %>%
  filter(img_set == 'train') %>%
  select(box_path_PB, species, location)
write.csv(train, 'data/kenya/train.csv')
test = tags_meta %>%
  filter(img_set == 'test') %>%
  select(box_path_PB, species, location)
write.csv(test, 'data/kenya/test.csv')
val = tags_meta %>%
  filter(img_set == 'val') %>%
  select(box_path_PB, species, location)
write.csv(val, 'data/kenya/val.csv')

# now write just box_path as a txt file
write.table(train$box_path_PB, 'data/kenya/train.txt', row.names = FALSE, col.names = FALSE, quote = FALSE)
write.table(test$box_path_PB, 'data/kenya/test.txt', row.names = FALSE, col.names = FALSE, quote = FALSE)
write.table(val$box_path_PB, 'data/kenya/val.txt', row.names = FALSE, col.names = FALSE, quote = FALSE)
# also need to do this on wild species only
domes = c('shoat', 'domestic_dog', 'cattle')
tags_wild = tags_meta %>%
  filter(!species %in% domes)

train_w = tags_wild %>%
  filter(img_set == 'train') %>%
  select(box_path_PB, species, location)
write.csv(train_w, 'data/kenya/train_wild.csv')
test_w = tags_wild %>%
  filter(img_set == 'test') %>%
  select(box_path_PB, species, location)
write.csv(test_w, 'data/kenya/test_wild.csv')
val_w = tags_wild %>%
  filter(img_set == 'val') %>%
  select(box_path_PB, species, location)
write.csv(val_w, 'data/kenya/val_wild.csv')

write.table(train_w$box_path_PB, 'data/kenya/train_wild.txt', row.names = FALSE, col.names = FALSE, quote = FALSE)
write.table(test_w$box_path_PB, 'data/kenya/test_wild.txt', row.names = FALSE, col.names = FALSE, quote = FALSE)
write.table(val_w$box_path_PB, 'data/kenya/val_wild.txt', row.names = FALSE, col.names = FALSE, quote = FALSE)



names(tags_meta$location) = 'ct_site'
write.csv(tags_meta, 'data/kenya/mmct_cropsmeta_PB.csv')
