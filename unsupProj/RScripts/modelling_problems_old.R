#modelling problems


# Data --------------------------------------------------------------------

library(ggplot2)
library(dplyr)
library(lme4)
library(ggcorrplot)
library(MuMIn)
library(VGAM)
oop = options(na.action= 'na.fail')
library(performance)
library(sjPlot)
source('unsupProj/cor_plot_func.R')

#np <- reticulate::import("numpy")

#load resnet supervised - HDBSCAN
#this is from 2-d umap, with settings on HDBSCAN to optimise silhouette score.
counts_RNS = read.csv('output/NP_RN50_full_cluster_count.csv')
plot(counts_RNS$distinct_label_count, counts_RNS$distinct_species_count)

#Load resnet unsupervised - HDBSCAN
counts_RNU = read.csv('output/PegNet_cluster_count.csv')
plot(counts_RNU$distinct_label_count, scale(counts_RNU$distinct_species_count))

#merge these dataframes
colnames(counts_RNS)[2] = 'imagescape_RNS'
colnames(counts_RNU)[2] = 'imagescape_RNU'
counts = cbind(counts_RNS, counts_RNU[2])

#scale variables
names(counts)

#load image meta data and filter
meta = read.csv('data/nepal_cropsmeta_PB.csv')
antro = c('human', 'vehicle')
meta = meta[!(meta$species %in% antro),]
meta = meta[meta$SetID=='train',]
covs = read.csv('data/NepalCovariates_2019LandCover.csv')

#create dataframe with all the variables - management, forestcover, 
head(counts)
imgs = meta %>% group_by(ct_site) %>% summarise(n_imgs = n())
counts$n_imgs = imgs$n_imgs

counts = left_join(counts, covs, by = c('X' = 'CT_site'))
head(counts)
names(counts)

#scale variables


# Scaled variables --------------------------------------------------------
#for poisson distribution, the count data should not be scaled or logged.
#https://besjournals.online.wiley.com/doi/full/10.1111/j.2041-210X.2010.00021.x

#so the imagescape diversity metrics are not included here. it is just for response variables
covs_num = counts[,c(3,15,16,18,35)] #species richness, distance to river, distace to roads, distance to village, forest cover
covs_num$distinct_species_count = as.numeric(covs_num$distinct_species_count)
covs_scaled = scale(covs_num)

scaled_df = cbind(counts[,c(1,2,4,5,7)], covs_scaled)
head(scaled_df)
scaled_df$Management = factor(scaled_df$Management, levels = c('BZ', 'NP', 'OBZ'))

scaled_df$imagescapeRNS_rel = scaled_df$imagescape_RNS/log(scaled_df$n_imgs)


# Correlation plot --------------------------------------------------------

#plot correlation of variables
covs_num= scaled_df[,c(2,3,4,6:11)]


#Question 1 - should n_imgs be logged to make the relative variable?
#Reason - when logged, the relationship between n_imgs and imagescape becomes more linear.
plot(covs_num$n_imgs,covs_num$imagescapeRNS_rel)
plot(log(log(covs_num$n_imgs)),covs_num$imagescapeRNS_rel)

head(covs_num)
corplot_covs<-covs_num %>% cor_plot() 
corplot_covs


plot(covs_num$n_imgs,covs_num$imagescapeRNS)
plot(log(log(covs_num$n_imgs)),covs_num$imagescapeRNS)

plot(covs_num$n_imgs,covs_num$imagescapeRNU)
plot(log(covs_num$n_imgs),covs_num$imagescapeRNU)

plot(scaled_df$imagescape_RNU,scaled_df$imagescapeRNS)
plot(log(log(covs_num$n_imgs)),covs_num$imagescapeRNS)


RNSi = ggplot(scaled_df, aes(x = imagescape_RNS, y = n_imgs)) +
  geom_point() +
  theme_bw() + 
  xlab('ImageScape Diversity - T-CNN') +
  ylab('Number of Images per CT Site')

RNSl = ggplot(scaled_df, aes(x = imagescape_RNS, y = log(n_imgs))) +
  geom_point() +
  theme_bw() + 
  xlab('ImageScape Diversity - T-CNN') +
  ylab('log(Number of Images per CT Site)')

RNUi = ggplot(scaled_df, aes(x = imagescape_RNU, y = n_imgs)) +
  geom_point() +
  theme_bw() + 
  xlab('ImageScape Diversity - P-CNN') +
  ylab('Number of Images per CT Site')

RNUl = ggplot(scaled_df, aes(x = imagescape_RNU, y = log(n_imgs))) +
  geom_point() +
  theme_bw() + 
  xlab('ImageScape Diversity - P-CNN') +
  ylab('log(Number of Images per CT Site)')

RNUi + RNUl + RNSi + RNSl

# Modelling ---------------------------------------------------------------


#Including offset or not?
m1.o = glm(imagescape_RNS ~ Management + distinct_species_count + forest500_19 + Management*distinct_species_count + distinct_species_count*forest500_19 + offset(log(log(n_imgs))), data = scaled_df, family = 'poisson')
summary(m1.o)
#here, including offset implies a continuous linear relationship between n imgs and metric. 
#double log in the offset means firstly accounting log-link in the model, second is converting imgs to log scale.

m1 = glm(imagescape_RNS ~ Management + distinct_species_count + forest500_19 + Management*distinct_species_count + distinct_species_count*forest500_19, data = scaled_df, family = 'poisson')


#or relative
m1.r = glm(imagescapeRNS_rel ~ Management + distinct_species_count + forest500_19 + DistRiver + Management*distinct_species_count + distinct_species_count*forest500_19, data = scaled_df, family = 'gaussian')

AIC(m1)
summary(m1.o)

m2.o = glm(imagescape_RNS ~ distinct_species_count + forest500_19 + distinct_species_count*forest500_19 + offset(log(log(n_imgs))), data = scaled_df, family = 'poisson')
summary(m2.o)
AIC(m1.o, m2.o)

m3.o = glm(imagescape_RNS ~ forest500_19 + offset(log(log(n_imgs))), data = scaled_df, family = 'poisson')
summary(m2.o)
AIC(m1.o, m2.o, m3.o)

m4.o = glm(imagescape_RNS ~ distinct_species_count + offset(log(log(n_imgs))), data = scaled_df, family = 'poisson')
summary(m4.o)
AIC(m1.o, m2.o, m3.o, m4.o)


#include plot of n_imgs log log 
#similarity to rareifaction

#when you've ended the model selection - run on species without scaled 


