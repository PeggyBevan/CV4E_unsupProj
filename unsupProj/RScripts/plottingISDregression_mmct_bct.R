# plotting ISD regression for mmct and bct.
# exploring clustering results 
library(ggplot2)
library(dplyr)
library(flextable)
library(patchwork)
library(lme4)
library(tidyr)
library(sjPlot)
library(performance) #check overdispersion
library(MASS) 
library(broom)
library(mgcv)
library(vegan)
library(tidyverse)#map_df

# bct data load
cbsbct = read.csv("results/nepal/clustercountsbysite.csv")#contains wild and non wild
regbct = read.csv("results/nepal/regression_summary.csv")

# the actual list of images that went into this
metabct = read.csv("data/nepal_cropsmeta_PB.csv") %>%
  filter(species != "human") %>%
  filter(species != "vehicle") %>%
  filter(SetID == 'train')
unique(metabct$species)
# what is the date range of this dataset?
range(metabct$date)
# there is one row that doesn't seem to have a date - remove this
metabct$datetime[metabct$date==''] = '2019-03-25 00:08:37'
metabct$date[metabct$date==''] = '2019-03-25'
metabct$time[metabct$time==''] = '00:08:37'
metabct$time_hour[metabct$time_hour==''] = 0
metabct$time_min[metabct$time_min==''] = 8
metabct$time_sec[metabct$time_sec==''] = 37

range(metabct$date)
# the range of the training data is from 2nd Feb to 25th May
class(metabct$date)
metabct$date = as.Date(metabct$date)
hist(metabct$date, breaks = "weeks")
# for now we will keep in all the data, but worth noting that the ferreira paper 
# used cutoff points of 15th March - 15th April. 
# plot regression results 

# mmct data load
cbsmmct = read.csv("results/kenya/clustercountsbysite.csv")#contains wild and non wild
regmmct = read.csv("results/kenya/regression_summary.csv")

metammct = read.csv("data/kenya/mmct_cropsmeta_PB.csv") %>%
  filter(img_set == 'train')
unique(metammct$species)
range(metammct$date)
# the range of the training data is 5th October 2018 to 10th November.
metammct$date = as.Date(metammct$date)
hist(metammct$date, breaks = 'weeks')

# merge results together
head(cbsbct)
head(cbsmmct)

# change 'PegNet' to 'resnet'
cbsbct$model = gsub("PegNet", "resnet", cbsbct$model)
cbsmmct$model = gsub("PegNet", "resnet", cbsmmct$model)
cbsbct$dataset = 'bct'
cbsmmct$dataset = 'mmct'
cbs = bind_rows(cbsbct, cbsmmct)

head(regbct)
head(regmmct)
regbct$model = gsub("PegNet", "resnet", regbct$model)
regmmct$model = gsub("PegNet", "resnet", regmmct$model)
regbct$dataset = 'bct'
regmmct$dataset = 'mmct'
reg = bind_rows(regbct, regmmct)

ggplot(cbs, aes(x = distinct_label_count, y = distinct_species_count, colour = as.factor(n_components),
                linetype = hdbscan_cfg)) +
  # geom_point(alpha = 0.1) +
  geom_smooth(method = "lm", se = FALSE) +
  facet_wrap(dataset~model, ncol = 2) +
  theme_bw() +
  ylim(0, 20) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey") +
  labs(x = "Cluster diversity", y = "Species Richness", colour = "UMAP Dimensions", linetype = "HDBSCAN Config") +
  theme(text = element_text(size = 12),
        strip.background = element_blank())

# just 2% min cluster size
ggplot(cbs[cbs$hdbscan_cfg=='leaf_2pct',], aes(x = distinct_label_count, y = distinct_species_count, colour = as.factor(n_components)
)) +
  geom_point(alpha = 0.2) +
  geom_smooth(method = "lm", se = FALSE) +
  facet_wrap(dataset~model) +
  ylim(0, 20) +
  theme_bw() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey") +
  labs(x = "ImageScape Diversity", y = "Species Richness", colour = "UMAP Dimensions", linetype = "HDBSCAN Config")

# just 0.5% min cluster size
ggplot(cbs[cbs$hdbscan_cfg=='leaf_0p5pct',], aes(x = distinct_label_count, y = distinct_species_count, colour = as.factor(n_components)
)) +
  geom_point(alpha = 0.2) +
  geom_smooth(method = "lm", se = FALSE) +
  facet_wrap(dataset~model) +
  ylim(0, 20) +
  theme_bw() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey") +
  labs(x = "ImageScape Diversity", y = "Species Richness", colour = "UMAP Dimensions", linetype = "HDBSCAN Config")

# just 1% min cluster size
ggplot(cbs[cbs$hdbscan_cfg=='leaf_1pct',], aes(x = distinct_label_count, y = distinct_species_count, colour = as.factor(n_components)
)) +
  geom_point(alpha = 0.2) +
  geom_smooth(method = "lm", se = FALSE) +
  facet_wrap(dataset~model) +
  ylim(0, 20) +
  theme_bw() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey") +
  labs(x = "ImageScape Diversity", y = "Species Richness", colour = "UMAP Dimensions", linetype = "HDBSCAN Config")

# make a table showing regression values - rows = model & n_dimensions,
# columns = regression values under different hdbscan condition

reg$min_cluster_size_pc = sub("leaf_", "", reg$hdbscan_cfg)
reg$min_cluster_size_pc = sub("pct", "%", reg$min_cluster_size_pc)
reg$min_cluster_size_pc = sub("p", ".", reg$min_cluster_size_pc)

reg$model = gsub("PegNet50", "resnet50", reg$model)
reg$wild = ifelse(grepl("wild", reg$model), "Wild species only", "All species")
reg$model_name = gsub("_wild", "", reg$model)

reg$model_name = gsub('_mmct', '', reg$model_name)
# make dataset all capitals
reg$dataset = toupper(reg$dataset)

# make a plot of this table
r2p = ggplot(reg, aes(x = as.factor(n_components), y = correlation_r, colour = model_name, shape = min_cluster_size_pc)) +
  geom_point()+
  facet_wrap(dataset~wild) +
  ylim(0, 1) +
  theme_bw() +
  scale_colour_viridis_d() +
  labs(x = "UMAP Dimensions", y = "Correlation (R2)", colour = "CNN Architecture", shape = 'Min cluster size') +
  theme(strip.background = element_blank(),
        strip.text = element_text(size = 12),
        text = element_text(size = 12))
r2p
np = ggplot(reg, aes(x = as.factor(n_components), y = noise_ratio_global, colour = model_name, shape = min_cluster_size_pc)) +
  geom_point()+
  facet_wrap(dataset~wild) +
  ylim(0, 1) +
  theme_bw() +
  labs(x = "UMAP Dimensions", y = "Noise Ratio", colour = "CNN Architecture", shape = 'Min cluster size') +
  scale_colour_viridis_d() +
  theme(strip.background = element_blank(),
        strip.text = element_text(size = 12),
        text = element_text(size = 12))
np

# total number of clusters
nc = ggplot(reg, aes( x = as.factor(n_components), y = n_clusters_global, 
                      colour = model_name, shape = min_cluster_size_pc)) +
  geom_point()+
  facet_wrap(dataset~wild) +
  theme_bw() +
  labs(x = "UMAP Dimensions", y = "Number of clusters", colour = "CNN Architecture", shape = 'Min cluster size') +
  scale_colour_viridis_d() +
  theme(strip.background = element_blank(),
        strip.text = element_text(size = 12),
        text = element_text(size = 12))
nc

r2p + np + nc + plot_layout(guide = "collect")


reg$wild = ifelse(grepl("wild", reg$model), "wild", "all")

# make a column for r2 all and r2 wild
reg_wide = pivot_wider(reg, 
                       id_cols = c("dataset", "model_name", "n_components","min_cluster_size_pc"), 
                       names_from = wild,
                       values_from = c('correlation_r', 'noise_ratio_global', "n_clusters_global"))
# select the highest r value for each model
bold_rows_all <- reg_wide %>%
  mutate(row_num = row_number()) %>%
  group_by(dataset, model_name) %>%
  slice_max(order_by = correlation_r_all, n = 1, with_ties = FALSE) %>%
  pull(row_num)
bold_rows_wild = reg_wide %>%
  mutate(row_num = row_number()) %>%
  group_by(model_name) %>%
  slice_max(order_by = correlation_r_wild, n = 1, with_ties = FALSE) %>%
  pull(row_num)

f1 = flextable(reg_wide)
# group by model and n_components
f1 = merge_v(f1, j = c(1,2)) %>%
  set_header_labels(dataset = 'Dataset', model_name = "Model", n_components = "UMAP Dimensions",
                    min_cluster_size_pc = "Min Cluster Size (%)",
                    correlation_r_all = "All species", 
                    correlation_r_wild = "Wild species only",
                    noise_ratio_global_all = 'All species',
                    noise_ratio_global_wild = 'Wild species only',
                    n_clusters_global_all = 'All species',
                    n_clusters_global_wild = 'Wild species only') %>%
  set_table_properties(layout = "autofit", width = .8)%>%
  hline(i = c(12, 24, 36), border = fp_border_default()) %>%
  bold(i = bold_rows_all, j = ~ correlation_r_all) %>%
  bold(i = bold_rows_wild, j = ~ correlation_r_wild) %>%
  add_header_row(values = c("", "R2", "Noise Ratio", 'Total clusters'), colwidths = c(4, 2,2,2))
f1

# save f1 as png
save_as_image(f1, "results/figures/regsum_full_mmct_bct.png", width = 8, height = 4, units = "in", res = 300)



# trying to select best model, using rankings
reg_wide1 <- reg_wide %>%
  group_by(dataset) %>%
  mutate(
    rank_corr  = rank(`correlation_r_wild`),
    rank_clust = rank(`n_clusters_global_wild`),
    rank_noise = rank(-`noise_ratio_global_wild`),  # negative so lower noise = higher rank
    rank_total = rank_corr + rank_clust + rank_noise
  ) %>%
  arrange(desc(rank_total))


# alternatively, lets choose the top 5 performing models and present these
reg_arr = arrange(reg_wide, desc(correlation_r_all))
f2_mmct  = reg_arr[,1:8] %>% filter(dataset=='MMCT') %>% slice(1:5) %>%
  flextable() %>%
  set_header_labels(dataset = 'Dataset', model_name = "Model", n_components = "UMAP Dimensions",
                    min_cluster_size_pc = "Min Cluster Size (%)",
                    correlation_r_all = "All species", 
                    correlation_r_wild = "Wild species only",
                    noise_ratio_global_all = 'All species',
                    noise_ratio_global_wild = 'Wild species only') %>%
  add_header_row(values = c("", "R2", "Noise Ratio"), colwidths = c(4, 2,2)) %>%
  set_table_properties(layout = "autofit", width = .8)
f2_mmct

f2_bct = reg_arr[,1:8] %>% filter(dataset=='BCT') %>% slice(1:5) %>%
  flextable() %>%
  set_header_labels(dataset = 'Dataset', model_name = "Model", n_components = "UMAP Dimensions",
                    min_cluster_size_pc = "Min Cluster Size (%)",
                    correlation_r_all = "All species", 
                    correlation_r_wild = "Wild species only",
                    noise_ratio_global_all = 'All species',
                    noise_ratio_global_wild = 'Wild species only') %>%
  add_header_row(values = c("", "R2", "Noise Ratio"), colwidths = c(4, 2,2)) %>%
  set_table_properties(layout = "autofit", width = .8)
f2_bct

save_as_image(f2_mmct, "results/figures/regsum_mmct_top5.png", width = 8,
              height = 4, units = "in", res = 300)
save_as_image(f2_bct, "results/figures/regsum_bct_top5.png", width = 8,
              height = 4, units = "in", res = 300)

# create same tables but ordered by correlations with wild species only
reg_arr = arrange(reg_wide, desc(correlation_r_wild))
f3_mmct  = reg_arr[,1:8] %>% filter(dataset=='MMCT') %>% slice(1:5) %>%
  flextable() %>%
  set_header_labels(dataset = 'Dataset', model_name = "Model", n_components = "UMAP Dimensions",
                    min_cluster_size_pc = "Min Cluster Size (%)",
                    correlation_r_all = "All species", 
                    correlation_r_wild = "Wild species only",
                    noise_ratio_global_all = 'All species',
                    noise_ratio_global_wild = 'Wild species only') %>%
  add_header_row(values = c("", "R2", "Noise Ratio"), colwidths = c(4, 2,2)) %>%
  set_table_properties(layout = "autofit", width = .8)
f3_mmct

f3_bct = reg_arr[,1:8] %>% filter(dataset=='BCT') %>% slice(1:5) %>%
  flextable() %>%
  set_header_labels(dataset = 'Dataset', model_name = "Model", n_components = "UMAP Dimensions",
                    min_cluster_size_pc = "Min Cluster Size (%)",
                    correlation_r_all = "All species", 
                    correlation_r_wild = "Wild species only",
                    noise_ratio_global_all = 'All species',
                    noise_ratio_global_wild = 'Wild species only') %>%
  add_header_row(values = c("", "R2", "Noise Ratio"), colwidths = c(4, 2,2)) %>%
  set_table_properties(layout = "autofit", width = .8)
f3_bct

save_as_image(f3_mmct, "results/figures/regsum_mmct_top5wild.png", width = 8, 
              height = 4, units = "in", res = 300)
save_as_image(f3_bct, "results/figures/regsum_bct_top5wild.png", width = 8,
              height = 4, units = "in", res = 300)

n_distinct(meta$species)

`# ANALYSIS PLAN

# Data prep 
# connect dataset with environmental variables of all the camera trap sites
# also need a dataset with each embedding vector connected to the image name, 
# so that time stamp can be extracted.

siteinfobct = read.csv("data/siteVariable_149sites_Nepal_Oct2020.csv")
siteinfommct = read.csv("data/kenya/KN_covariates_2018_livestockrates_v3.csv")


cbsbct1 = left_join(cbsbct, siteinfobct, by = c("ct_site" = "CT_site"))
cbsmmct1 = left_join(cbsmmct, siteinfommct, by = c("ct_site" = "CT_site"))
cbsmmct1$model = gsub("_mmct", "", cbsmmct1$model)
# need to get a measure of survey effort - number of camera trap days for each CT site.
sebct = read.csv("data/SurveyEffort_Mar-Apr_2019_corrected.csv")
semmct = read.csv("data/kenya/KN_all_sites_effort_occupancy_v2_2018.csv")

range(metabct$datetime)
range(sebct$X)

nrow(metabct[metabct$datetime < '2019-03-15',])
nrow(metabct)
# we cannot get full number of camera trap days for nepal because my survey effort table is
# only between 15th March - 15th April.
# cutting the meta data down to fit SE will only remove 754/28605 images

range(metammct$datetime)
range(semmct$Date)
# here the survey effort goes beyond our available date, likely due to training data being split by time


# 1. ISD AS AN ECOSYSTEM CONDITION INDEX
# used glms or gam to predict ISD over habitat gradient. compare coefficients with 
# model from species richness
# will need to include survey effort.

# first, lets plot species richness across land-use types

# subset to one model type - the once that has highest performance in f2
# convnextL wild with leaf_1% and 8 umap dimensions
# this has a correlation of 0.74 & noise ratio of 0.53
topcbsbct = cbsbct1 %>%
  filter(model == "convnextL_wild") %>%
  filter(n_components == 8) %>%
  filter(hdbscan_cfg == "leaf_1pct")

topcbsbct$Management = factor(topcbsbct$Management, levels = c("NP", "BZ", "OBZ"))

a = ggplot(data = topcbsbct, aes(x = distinct_label_count, y = distinct_species_count)) +
  geom_point() +
  geom_smooth(method = "lm") +
  theme_bw() +
  labs(x = "VOTU Diversity", y = "Species Richness") +
  annotate("text", x = 10, y = 14, label = "R2 = 0.74", size = 6) +
  
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey") +
  ggtitle("a)") +
  ylim(0, 15) +
  theme(text = element_text(size = 16))
a
# species richness & ISD over management zone
b = ggplot(data = topcbsbct, aes(x = Management, y = distinct_species_count)) +
  geom_boxplot() +
  theme_bw() +
  labs(x = "Management Zone", y = "Species Richness") +
  ggtitle("b)") +
  theme(text = element_text(size = 16))
b
c= ggplot(data = topcbsbct, aes(x = Management, y = distinct_label_count)) +
  geom_boxplot() +
  theme_bw() +
  labs(x = "Management Zone", y = "VOTU diversity") +
  ggtitle("c)") +
  theme(text = element_text(size = 16))
c
a+b+c

# species richness over forest cover

d = ggplot(data = topcbsbct, aes(x = propForest500, y = distinct_species_count)) +
  geom_point() +
  geom_smooth(method = "lm") +
  theme_bw() +
  labs(x = "Proportion Forest Cover (500m radius)", y = "Species Richness") +
  ggtitle("d)") +
  theme(text = element_text(size = 16))

e = ggplot(data = topcbsbct, aes(x = propForest500, y = distinct_label_count)) +
  geom_point() +
  geom_smooth(method = "lm") +
  theme_bw() +
  labs(x = "Proportion Forest Cover (500m radius)", y = "VOTU Diversity") +
  ggtitle("e)") +
  theme(text = element_text(size = 16))


a+plot_spacer()+b+c+d+e +plot_layout(ncol = 2)


# originally i correlated ISD with species richness
domes = c('buffalo', 'cow', 'dog', 'domestic_cat', 'domestic_chicken', 'domestic elephant', 'goat', 'sheep')

# kenya
# the same model is also very high performing for kenya - it seems to have a comparativly low noise ratio & a 0.78 R2 value with species richness.
topcbsmmct = cbsmmct1 %>%
  filter(model == "convnextL_wild") %>%
  filter(n_components == 8) %>%
  filter(hdbscan_cfg == "leaf_1pct")
# MT = mara triangle
# MN = mara north conservancy
# OM =  Olare-Motorogi
# NB = Naboisho Conservancy
# there are some sites that can be removed due to low survey effort
topcbsmmct = topcbsmmct %>%
  filter(conservancy != "")

topcbsmmct$conservancy = factor(topcbsmmct$conservancy, levels = c("MT","MN", "OM", "NB"))

a = ggplot(data = topcbsmmct, aes(x = distinct_label_count, y = distinct_species_count)) +
  geom_point() +
  geom_smooth(method = "lm") +
  theme_bw() +
  labs(x = "VOTU Diversity", y = "Species Richness") +
  annotate("text", x = 10, y = 25, label = "R2 = 0.78", size = 6) +
  
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey") +
  ggtitle("a)") +
  ylim(0, 27) +
  theme(text = element_text(size = 16))
a
# species richness & ISD over management zone
b = ggplot(data = topcbsmmct, aes(x = conservancy, y = distinct_species_count)) +
  geom_boxplot() +
  theme_bw() +
  labs(x = "Conservancy", y = "Species Richness") +
  ggtitle("b)") +
  theme(text = element_text(size = 16))
b
c= ggplot(data = topcbsmmct, aes(x = conservancy, y = distinct_label_count)) +
  geom_boxplot() +
  theme_bw() +
  labs(x = "Conservancy", y = "VOTU Diversity") +
  ggtitle("c)") +
  theme(text = element_text(size = 16))
c
a+b+c

# species richness over dist from human settlement
# are there any variables that have a negative relationship with species richness?
target = topcbsmmct[,8]
features = topcbsmmct[,c(16:22,24:38,40:50)]
cor_results <- cor(features, target, use = "pairwise.complete.obs")

# NDVI and SAVI are very slightly neg correlated.
d = ggplot(data = topcbsmmct, aes(x = Mean.savi, y = distinct_species_count)) +
  geom_point() +
  geom_smooth(method = "lm") +
  theme_bw() +
  labs(x = "Soil-Adjusted Vegetation Index (SAVI)", y = "Species Richness") +
  ggtitle("d)") +
  theme(text = element_text(size = 16))
d
e = ggplot(data = topcbsmmct, aes(x = Mean.savi, y = distinct_label_count)) +
  geom_point() +
  geom_smooth(method = "lm") +
  theme_bw() +
  labs(x = "Soil-Adjusted Vegetation Index (SAVI)", y = "VOTU Diversity") +
  ggtitle("e)") +
  theme(text = element_text(size = 16))

a+plot_spacer()+b+c+d+e +plot_layout(ncol = 2)

# Linear modelling of ISD -------------------------------------------------

# terms all need to be scaled so they match - species richness, forest cover and image count

# i need to measure survey effort properly. 
# realistically i need to only use images from the survey period as the
# images from random sampling are not systematically sampled, 
# so will not be related to survey effort. 
# this needs to happen in the python script because the data i have 
# is already summarised.
# for now, just use the survey effort i have

# bct
ctdaysbct = sebct[,-1] %>%
  colSums(na.rm = TRUE) %>%
  data.frame(ct_site = names(.), camtrapdays = .)

topcbsbct = topcbsbct %>%
  left_join(ctdaysbct, by = "ct_site")
# scale the forest cover variable
topcbsbct$propForest500_scaled = as.numeric(scale(topcbsbct$propForest500))
topcbsbct$log_camtrapdays = log(topcbsbct$camtrapdays)
topcbsbct$Management = factor(topcbsbct$Management, levels = c("NP", "BZ", "OBZ"))

# mmct
range(metammct$datetime)
range(semmct$Date)
# cut off survey effort after 10th November 2018, which is the last date of the training data
semmct$Date = as.Date(semmct$Date, format = "%Y.%m.%d")
semmct = semmct %>%
  filter(Date <= '2018-11-10')
ctdaysmmct = semmct[,-1] %>%
  colSums(na.rm = TRUE) %>%
  data.frame(ct_site = names(.), camtrapdays = .)

topcbsmmct = topcbsmmct %>%
  left_join(ctdaysmmct, by = "ct_site")
topcbsmmct$Mean.savi_scaled = as.numeric(scale(topcbsmmct$Mean.savi))
topcbsmmct$humdist_short_scaled = as.numeric(scale(topcbsmmct$humdist_short))
topcbsmmct$propopen500m_scaled = as.numeric(scale(topcbsmmct$propopen500m))
topcbsmmct$livestock_avg_500m_scaled = as.numeric(scale(topcbsmmct$livestock_avg_500m))
topcbsmmct$shoat_30min_event_rate_scaled = as.numeric(scale(topcbsmmct$shoat_30min_event_rate))
topcbsmmct$cattle_30min_event_rate_scaled = as.numeric(scale(topcbsmmct$cattle_30min_event_rate))
topcbsmmct$log_camtrapdays = log(topcbsmmct$camtrapdays)
topcbsmmct$conservancy = factor(topcbsmmct$conservancy, levels = c("MT","MN", "OM", "NB"))

# To do: re-run models or get regression coefficients/label counts on 
# images only from 15th March - 15th April. 

m5_sr = glm(distinct_species_count ~ Management*propForest500_scaled - 1 + 
              offset(log_camtrapdays), 
            data = topcbsbct, family = "poisson")
check_overdispersion(m5_sr)
# no overdispersion here, so we can use poisson distribution.
summary(m5_sr)

m5_isd = glm(distinct_label_count ~ Management*propForest500_scaled -1 +
               offset(log_camtrapdays), 
             data = topcbsbct, family = "poisson")
summary(m5_isd)
check_overdispersion(m5_isd)
# overdispersion detected, but it just on the limit
m5_isdnb = glm.nb(distinct_label_count ~ Management + propForest500_scaled + 
                    offset(log_camtrapdays), 
                  data = topcbsbct)
summary(m5_isdnb)
# nb can't converge
# we can try quasipoisson instead
m5_isdqp = glm(distinct_label_count ~ Management * propForest500_scaled - 1  + 
                 offset(log_camtrapdays), 
               data = topcbsbct, family = quasipoisson())
summary(m5_isdqp)
# this does not give an AIC. It is possible to compare coefficients with the poisson model, 
# but bare in mind that the standard error will be wider for the quasipoisson model, so the confidence intervals will be wider and there may be less significant results.

# compare model coefficients
sr_coefs <- tidy(m5_sr, conf.int = TRUE, exponentiate = TRUE) %>% 
  mutate(response = "Species Richness")
isd_coefs <- tidy(m5_isdqp, conf.int = TRUE, exponentiate = TRUE) %>% 
  mutate(response = "VOTU Diversity")

df1 = bind_rows(sr_coefs, isd_coefs) %>%
  mutate(term = recode(term,
                       "ManagementNP" = "Low disturbance",
                       "ManagementBZ" = "Medium disturbance",
                       "ManagementOBZ" = "High disturbance",
                       "propForest500_scaled" = "Forest Cover (500m)",
                       "ManagementOBZ:propForest500_scaled" = "High disturbance:Forest Cover",
                       "ManagementBZ:propForest500_scaled" = "Med disturbance:Forest Cover"))
df1$term = factor(df1$term,levels = c("Low disturbance",
                                      "Medium disturbance",
                                      "High disturbance",
                                      "Forest Cover (500m)",
                                      "Med disturbance:Forest Cover",
                                      "High disturbance:Forest Cover"))

coefp = ggplot(df1, aes(x = estimate, y = term, colour = response,
                        xmin = conf.low, xmax = conf.high)) +
  geom_pointrange(position = position_dodge(width = 0.4)) +
  geom_vline(xintercept = 1, linetype = "dashed", colour = "grey50") +
  scale_colour_manual(values = c("Species Richness" = "#2166ac", 
                                 "VOTU Diversity" = "#d6604d")) +
  labs(x = "Rate Ratio", y = NULL, colour = NULL) +
  theme_classic() +
  theme(text = element_text(size = 16)) +
  ggtitle("a)")
coefp
library(ggeffects)

pred_mgmt <- ggpredict(m5_isdqp, terms = "Management", 
                       condition = c(camtrapdays = mean(topcbsbct$log_camtrapdays)))

mgmtisd = ggplot() +
  geom_jitter(data = topcbsbct, aes(x = Management, y = distinct_label_count),
              width = 0.15, alpha = 0.4) +
  geom_pointrange(data = pred_mgmt, 
                  aes(x = x, y = predicted, ymin = conf.low, ymax = conf.high),
                  colour = "#d6604d", size = 1) +
  labs(x = "Management Zone", y = "Cluster Diversity") +
  theme_classic() +
  theme(text = element_text(size = 16)) +
  ggtitle("b)")
mgmtisd
pred_mgmt_sr<- ggpredict(m5_sr, terms = "Management", 
                         condition = c(camtrapdays = mean(topcbsbct$log_camtrapdays)))

mgmtsr = ggplot() +
  geom_jitter(data = topcbsbct, aes(x = Management, y = distinct_species_count),
              width = 0.15, alpha = 0.4) +
  geom_pointrange(data = pred_mgmt_sr, 
                  aes(x = x, y = predicted, ymin = conf.low, ymax = conf.high),
                  colour = "#d6604d", size = 1) +
  labs(x = "Management Zone", y = "Species Richness") +
  theme_classic() +
  theme(text = element_text(size = 16)) +
  ggtitle("a)")

mgmtsr + mgmtisd

# plot the same for forest cover across management regime.
forest_mean <- mean(topcbsbct$propForest500, na.rm = TRUE)
forest_sd <- sd(topcbsbct$propForest500, na.rm = TRUE)

hist(topcbsbct$propForest500[topcbsbct$Management == "NP"])
hist(topcbsbct$propForest500[topcbsbct$Management == "BZ"])
hist(topcbsbct$propForest500[topcbsbct$Management == "OBZ"])
# there is little variation in forest cover in NP, so we won't predict for this group

pred_forest <- ggpredict(m5_isdqp, 
                         terms = c("propForest500_scaled [n = 100]", "Management"),
                         condition = c(camtrapdays = mean(topcbsbct$log_camtrapdays)))
pred_forest$group = recode(pred_forest$group,
                           "NP" = "Low",
                           "BZ" = "Medium",
                           "OBZ" = "High")
for_isd = ggplot(pred_forest[pred_forest$group!='Low',], 
                 aes(x = x, y = predicted, colour = group, fill = group)) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = 0.2, colour = NA) +
  geom_line(size = 1) +
  scale_x_continuous(
    breaks = (c(0, 0.25, 0.5, 0.75, 1) - forest_mean) / forest_sd,
    labels = c("0", "0.25", "0.5", "0.75", "1")
  ) +
  labs(x = "Forest Cover (500m, proportion)", y = "VOTU Diversity", 
       colour = "Disturbance level", fill = 'Disturbance level') +
  theme_classic() +
  theme(text = element_text(size = 12))

pred_forest_sr <- ggpredict(m5_sr, 
                            terms = c("propForest500_scaled [n = 100]", "Management"),
                            condition = c(camtrapdays = mean(topcbsbct$log_camtrapdays)))


for_sr = ggplot(pred_forest_sr[pred_forest_sr$group!='NP',], 
                aes(x = x, y = predicted, colour = group, fill = group)) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = 0.2, colour = NA) +
  geom_line(size = 1) +
  scale_x_continuous(
    breaks = (c(0, 0.25, 0.5, 0.75, 1) - forest_mean) / forest_sd,
    labels = c("0", "0.25", "0.5", "0.75", "1")
  ) +
  # scale_x_continuous(
  #   labels = function(x) round(x * forest_sd + forest_mean, 2)
  # ) +
  labs(x = "Forest Cover (500m, proportion)", y = "Species Richness", 
       colour = "Disturbance level", fill = 'Disturbance level') +
  theme_classic() +
  theme(text = element_text(size = 12),
        legend.position = "none") +
  ggtitle("b)")

for_sr + for_isd & plot_layout(guides = "collect")

coefp
for_sr + for_isd)

# MMCT Linear modelling

m1_sr = glm(distinct_species_count ~ conservancy + humdist_short_scaled + 
              Mean.savi_scaled + propopen500m_scaled +
              livestock_avg_500m_scaled + shoat_30min_event_rate_scaled + 
              cattle_30min_event_rate_scaled - 1 + 
              offset(log_camtrapdays), 
            data = topcbsmmct, family = "poisson")
check_overdispersion(m1_sr)
# overdispersion here 
m1_srnb = glm.nb(distinct_species_count ~ conservancy + 
                   humdist_short_scaled + Mean.savi_scaled + propopen500m_scaled +
                    livestock_avg_500m_scaled + shoat_30min_event_rate_scaled + 
                    cattle_30min_event_rate_scaled - 1 + 
                    offset(log_camtrapdays), 
                  data = topcbsmmct)
check_overdispersion(m1_srnb)
summary(m1_srnb)

m2_isd = glm(distinct_label_count ~ conservancy + 
               humdist_short_scaled + Mean.savi_scaled + propopen500m_scaled +
               livestock_avg_500m_scaled + shoat_30min_event_rate_scaled + 
               cattle_30min_event_rate_scaled - 1 + 
               offset(log_camtrapdays), 
             data = topcbsmmct, family = "poisson")

check_overdispersion(m2_isd)
# overdispersion detected, but it just on the limit
m2_isdnb = glm.nb(distinct_label_count ~ conservancy + 
                    humdist_short_scaled + Mean.savi_scaled + propopen500m_scaled +
                    livestock_avg_500m_scaled + shoat_30min_event_rate_scaled + 
                    cattle_30min_event_rate_scaled - 1 + 
                    offset(log_camtrapdays), 
                  data = topcbsmmct)
check_overdispersion(m2_isdnb)
summary(m2_isdnb)

# compare model coefficients
sr_coefs_mm <- tidy(m1_srnb, conf.int = TRUE, exponentiate = TRUE) %>% 
  mutate(response = "Species Richness")
isd_coefs_mm <- tidy(m2_isdnb, conf.int = TRUE, exponentiate = TRUE) %>% 
  mutate(response = "VOTU Diversity")

df1mm = bind_rows(sr_coefs_mm, isd_coefs_mm) %>%
  mutate(term = recode(term,
                       "conservancyMT" = "Mara Triangle NP",
                       "conservancyMN" = "Mara North",
                       "conservancyNB" = "Naboisho",
                       "conservancyOM" = "Olare-Motorogi",
                       "humdist_short_scaled" = "Dist to infrastructure",
                       "Mean.savi_scaled" = "SAVI",
                       "propopen500m_scaled" = "Open habitat (500m)",
                       "livestock_avg_500m_scaled" = "Livestock density (500m)",
                       "shoat_30min_event_rate_scaled" = "Shoat density",
                       "cattle_30min_event_rate_scaled" = "Cattle density"))
df1mm$term = factor(df1mm$term,levels = c("Mara Triangle NP",
                                      "Mara North",
                                      "Olare-Motorogi",
                                      "Naboisho",
                                      "Dist to infrastructure",
                                      "SAVI",
                                      "Open habitat (500m)",
                                      "Livestock density (500m)",
                                      "Shoat density",
                                      "Cattle density"))

coefpmm = ggplot(df1mm, aes(x = estimate, y = term, colour = response,
                        xmin = conf.low, xmax = conf.high)) +
  geom_pointrange(position = position_dodge(width = 0.4)) +
  geom_vline(xintercept = 1, linetype = "dashed", colour = "grey50") +
  scale_colour_manual(values = c("Species Richness" = "#2166ac", 
                                 "VOTU Diversity" = "#d6604d")) +
  labs(x = "Rate Ratio", y = NULL, colour = NULL) +
  theme_classic() +
  theme(text = element_text(size = 16)) +
  ggtitle("a)")
coefpmm

pred_cons <- ggpredict(m2_isdnb, terms = "conservancy", 
                       condition = c(camtrapdays = mean(topcbsbct$log_camtrapdays)))

consisd = ggplot() +
  geom_jitter(data = topcbsmmct, aes(x = conservancy, y = distinct_label_count),
              width = 0.15, alpha = 0.4) +
  geom_pointrange(data = pred_cons, 
                  aes(x = x, y = predicted, ymin = conf.low, ymax = conf.high),
                  colour = "#d6604d", size = 1) +
  labs(x = "Conservancy", y = "VOTU Diversity") +
  theme_classic() +
  theme(text = element_text(size = 16)) +
  ggtitle("b)") +
  ylim(0,27)
consisd
pred_cons_sr<- ggpredict(m1_srnb, terms = "conservancy", 
                         condition = c(camtrapdays = mean(topcbsbct$log_camtrapdays)))

conssr = ggplot() +
  geom_jitter(data = topcbsmmct, aes(x = conservancy, y = distinct_species_count),
              width = 0.15, alpha = 0.4) +
  geom_pointrange(data = pred_cons_sr, 
                  aes(x = x, y = predicted, ymin = conf.low, ymax = conf.high),
                  colour = "#d6604d", size = 1) +
  labs(x = "Conservancy", y = "Species Richness") +
  theme_classic() +
  theme(text = element_text(size = 16)) +
  ggtitle("a)")

conssr + consisd

# plot the same for shoat density (across conservancy?)
shoat_mean <- mean(topcbsmmct$shoat_30min_event_rate, na.rm = TRUE)
shoat_sd <- sd(topcbsmmct$shoat_30min_event_rate, na.rm = TRUE)

hist(topcbsmmct$shoat_30min_event_rate[topcbsmmct$conservancy == "MT"])
hist(topcbsmmct$shoat_30min_event_rate[topcbsmmct$conservancy == "MN"])
hist(topcbsmmct$shoat_30min_event_rate[topcbsmmct$conservancy == "OM"])
hist(topcbsmmct$shoat_30min_event_rate[topcbsmmct$conservancy == "NB"])

# there is no shoat density in MT, so we won't predict for this group

pred_shoat <- ggpredict(m2_isdnb, 
                         terms = c("shoat_30min_event_rate_scaled [n = 100]", "conservancy"),
                         condition = c(camtrapdays = mean(topcbsmmct$log_camtrapdays)))

pred_shoat$group = recode(pred_shoat$group,
                          "MT" = "Mara Triangle NP",
                          "MN" = "Mara North",
                          "OM" = "Olare-Motorogi",
                          "NB" = "Naboisho")
unscaled_ticks <- c(0, 50, 100, 150, 200)
shoat_isd = ggplot(pred_shoat[pred_shoat$group!='Mara Triangle NP',], 
                 aes(x = x, y = predicted, colour = group, fill = group)) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = 0.2, colour = NA) +
  geom_line(size = 1) +
  scale_x_continuous(
    breaks = (unscaled_ticks - shoat_mean) / shoat_sd,
    labels = as.character(unscaled_ticks)
    ) +
  labs(x = "Shoat Density", y = "VOTU Diversity", 
       colour = "Conservancy", fill = 'Conservancy') +
  theme_classic() +
  theme(text = element_text(size = 12)) +
  ylim(0,20)

pred_shoat_sr <- ggpredict(m1_srnb, 
                            terms = c("shoat_30min_event_rate_scaled [n = 100]", "conservancy"),
                            condition = c(camtrapdays = mean(topcbsmmct$log_camtrapdays)))

shoat_sr = ggplot(pred_shoat_sr[pred_shoat_sr$group!='MT',], 
                aes(x = x, y = predicted, colour = group, fill = group)) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = 0.2, colour = NA) +
  geom_line(size = 1) +
  scale_x_continuous(
    breaks = (unscaled_ticks - shoat_mean) / shoat_sd,
    labels = as.character(unscaled_ticks)
  ) +
  # scale_x_continuous(
  #   labels = function(x) round(x * forest_sd + forest_mean, 2)
  # ) +
  labs(x = "Shoat Density", y = "Species Richness", 
       colour = "Conservancy", fill = 'Conservancy') +
  theme_classic() +
  theme(text = element_text(size = 12),
        legend.position = "none") +
  ggtitle("b)") +
  ylim(0,20)

# shoat_sr + shoat_isd & plot_layout(guides = "collect")

coefp
shoat_sr + shoat_isd


# 2. ISD for measuring change in community
# compute pairwise dissimilarity across sites (NP vs BZ vs OBZ) and measure 
# community change across the pressure gradient. compare with actual species turnover

clustID = read.csv('results/image_cluster_labels_ConvNeXtL_wild_umap8_leaf_1pct.csv')
nrow(meta[!(meta$species %in% domes),])
unique(clustID$cluster_label)
unique(clustID$species)
# we can remove all images that were labelled as noise
noise = clustID %>%
  filter(cluster_label == -1)

clustID = clustID %>%
  filter(cluster_label != -1)

clustID = left_join(clustID, siteinfo, by = c("ct_site" = "CT_site"))
clustID = left_join(clustID, camtrapdays, by = "ct_site")

# change management to disturbance level low medium and high
clustID = clustID %>%
  mutate(disturbance = case_when(
    Management == "NP" ~ "Low",
    Management == "BZ" ~ "Medium",
    Management == "OBZ" ~ "High"
  ))

meta = meta %>%
  mutate(disturbance = case_when(
    conservancy_name == "NP" ~ "Low",
    conservancy_name == "BZ" ~ "Medium",
    conservancy_name == "OBZ" ~ "High"
  )) %>%
  left_join(camtrapdays, by = c("ct_site" = "ct_site"))
# Compositional dissimilarity across the gradient -------------------------

# Build a site x cluster matrix (proportional abundance of each cluster at each site)
cluster_matrix <- clustID %>%
  group_by(disturbance, ct_site, camtrapdays, cluster_label) %>%
  summarise(n = n()) %>%
  mutate(rate = n / camtrapdays) %>% #standardise by effort
  pivot_wider(id_cols = c(disturbance, ct_site),
              names_from = cluster_label, 
              values_from = rate,
              values_fill = 0)

# Remove empty rows
clustmat =  as.matrix(cluster_matrix[, -c(1:2)])
nonzero <- rowSums(clustmat, na.rm = TRUE) > 0
cluster_matrix = cluster_matrix[nonzero, ]
clustmat <- clustmat[nonzero, ]

clust_nmds = {
  # transform to hellinger
  clustmat_hell = decostand(clustmat, method = 'hellinger')
  nmds <- metaMDS(clustmat_hell, distance = "euclidian", k = 2, trymax = 100)
  # nmds_bray = metaMDS(clustmat, distance = "bray", k = 2, trymax = 100)
  scores_df <- as.data.frame(scores(nmds, display = 'sites'))
  scores_df$disturbance <- factor(cluster_matrix$disturbance, 
                                  levels = c("Low", "Medium", "High"))
  # scores_bray = as.data.frame(scores(nmds_bray, display = 'sites'))
  # scores_bray$disturbance <- factor(cluster_matrix$disturbance, 
  #                                 levels = c("Low", "Medium", "High"))
  
  # # Ellipse generator
  # veganCovEllipse <- function(cov, center = c(0,0), scale = 1, npoints = 100) {
  #   theta <- seq(0, 2 * pi, length.out = npoints)
  #   circle <- cbind(cos(theta), sin(theta))
  #   t(center + scale * t(circle %*% chol(cov, pivot = TRUE)))
  # }
  # 
  # ellipses <- map_df(levels(scores_df$Management), function(g) {
  #   sub <- scores_df[scores_df$Management == g, ]
  #   if (nrow(sub) < 3) return(NULL)
  #   ell <- with(sub, veganCovEllipse(
  #     cov.wt(cbind(NMDS1, NMDS2), wt = rep(1/nrow(sub), nrow(sub)))$cov,
  #     center = colMeans(cbind(NMDS1, NMDS2))))
  #   cbind(as.data.frame(ell), Group = g)
  # })
  # ellipses$Group = factor(ellipses$Group, levels = c("NP", "BZ", "OBZ"))
  # Plot
  clust_hell = ggplot(data = scores_df, aes(x = NMDS1, y = NMDS2)) + 
    geom_point(aes(color = disturbance), size = 3, alpha = 0.5) +
    stat_ellipse(level = 0.95, aes(color = disturbance), size = 1) +
    labs(title = "b) VOTU composition", colour = "Disturbance level") +
    annotate("text", x = 0.4, y = 1, label = paste0("Stress = ", round(nmds$stress, 3)), size = 4) +
    theme_bw()
}

# clust_bray = ggplot(data = scores_bray, aes(x = NMDS1, y = NMDS2)) + 
#   geom_point(aes(color = disturbance), size = 3, alpha = 0.5) +
#   stat_ellipse(level = 0.95, aes(color = disturbance), size = 1) +
#   labs(title = "Cluster composition (Bray-Curtis)", colour = "Disturbance level") +
#   annotate("text", x = 0.7, y = 1.2, label = paste0("Stress = ", round(nmds_bray$stress, 3)), size = 5)

# clust_hell + clust_bray & plot_layout(guides = "collect") 

# scores_df$ct_site <- cluster_matrix$ct_site[nonzero]
# # find the outlier - large negative NMDS1 & management = NP
# scores_df %>% filter(NMDS1 < -0.4)
# # NP28.
# scores_df %>% filter(NMDS2 < -0.5)
# # OBZ45
# np28 = meta %>% filter(ct_site == "NP28")
# obz45 = meta %>% filter(ct_site == "OBZ45")
# obz22 = meta %>% filter(ct_site == "OBZ22")
# np23 = meta %>% filter(ct_site == "NP23")

# np28 is mainly sloth bear and no chital
# OBZ45 has a lot more dog than usual.

spec_nmds = {
  species_matrix = meta %>%
    filter(!(species %in% domes)) %>%
    group_by(disturbance, ct_site, camtrapdays, species) %>%
    summarise(n = n()) %>%
    mutate(rate = n / camtrapdays) %>% #standardise by effort
    pivot_wider(id_cols = c(disturbance, ct_site),
                names_from = species, 
                values_from = rate,
                values_fill = 0)
  
  specmat = as.matrix(species_matrix[, -c(1:2)])
  nonzero <- rowSums(specmat, na.rm = TRUE) > 0
  species_matrix = species_matrix[nonzero, ]
  specmat <- specmat[nonzero, ]
  
  # transform to hellinger
  specmat_hell = decostand(specmat, method = 'hellinger')
  nmds <- metaMDS(specmat_hell, distance = "euclidian", k = 2, trymax = 100)
  scores_df <- as.data.frame(scores(nmds, display = 'sites'))
  scores_df$disturbance <- factor(species_matrix$disturbance, levels = c("Low", "Medium", "High"))
  
  ggplot(data = scores_df, aes(x = NMDS1, y = NMDS2)) + 
    geom_point(aes(color = disturbance), size = 3, alpha = 0.5) +
    stat_ellipse(level = 0.95, aes(color = disturbance), size = 1) +
    labs(title = "a) Species composition", colour = "Disturbance level") +
    annotate("text", x = 0.4, y = 1, label = paste0("Stress = ", round(nmds$stress, 3)), size = 4) +
    theme_bw() +
    theme(legend.position = "none")
  
}
spec_nmds + clust_nmds & plot_layout(guides = "collect")

# TO DO: can't do this test if the dimensions are different.
# compute dissimilarity matrix
# Bray-Curtis dissimilarity (accounts for abundance, not just presence)
dist_isd <- dist(decostand(clustmat, method = 'hellinger'))
# Also compute true species-based dissimilarity for comparison
dist_sp <- dist(decostand(specmat, method = 'hellinger'))

mantel(dist_isd, dist_sp, method = "pearson", permutations = 999)
# A mantel test is significant (P = 0.001) with a correlation of 0.62, 
# indicating a strong relationship between the VOTU-based dissimilarity 
# and the species-based dissimilarity across sites. 
# This suggests that the VOTUs are capturing meaningful ecological 
# differences between sites that are also reflected in the species composition.

# betadisper
bd <- betadisper(dist_isd, cluster_matrix$disturbance)
anova(bd)
# p = 0.07
bd_sp = betadisper(dist_sp, species_matrix$disturbance)
anova(bd_sp)
# p = 0,00014

# permanova
adonis2(dist_isd ~ disturbance, data = cluster_matrix, permutations = 999)
# p = 0.001,
adonis2(dist_sp ~ disturbance, data = species_matrix, permutations = 999)
# the permanova
# 0.001


# Biodiversity Intactness Index -------------------------------------------


# Abundance component
abundance_by_zone <- function(data, taxon_col) {
  data %>%
    group_by(disturbance, ct_site, .data[[taxon_col]]) %>%
    summarise(n = n(), .groups = "drop") %>%
    group_by(disturbance, ct_site) %>%
    summarise(total_det = sum(n), .groups = "drop") %>%
    mutate(det_rate = total_det / camtrapdays) %>%
    group_by(disturbance) %>%
    summarise(mean_rate = mean(det_rate), .groups = "drop") %>%
    mutate(
      NP_rate = mean_rate[disturbance == "Low"],
      abundance_BII = mean_rate / NP_rate
    )
}
# at the moment this doesn't work because noise points have been removed that would be in the OG dataset..
# ab_species <- abundance_by_zone(clustID, "species")
# ab_cluster <- abundance_by_zone(clustID, "cluster_label")

abundance_by_site <- function(data) {
  site_rates = data %>%
    group_by(disturbance, ct_site, camtrapdays) %>% 
    summarise(total_det = n(), .groups = "drop") %>% # count total detections per site
    mutate(det_rate = total_det / camtrapdays)
  
  NP_rate = site_rates %>%
    filter(disturbance == "Low") %>%
    summarise(mean_rate = mean(det_rate)) %>%
    pull(mean_rate)
  
  site_rates %>%
    mutate(abundance_BII = det_rate / NP_rate)
  
}

ab_cluster = abundance_by_site(clustID)
ab_species = abundance_by_site(meta[!(meta$species %in% domes),])

# ── 2. COMPOSITIONAL SIMILARITY COMPONENT ────────────────────────────────────
# Build site x taxon matrix, then calculate balanced Bray-Curtis similarity
# of each BZ/OBZ site against the average NP community
library(betapart)
comp_similarity <- function(data, taxon_col) {
  
  # site x taxon count matrix
  mat <- data %>%
    group_by(ct_site, disturbance, camtrapdays, .data[[taxon_col]]) %>%
    summarise(n = n(), .groups = "drop") %>%
    mutate(det_rate = n / camtrapdays) %>% # standardise by effort
    pivot_wider(names_from = all_of(taxon_col), 
                values_from = det_rate, values_fill = 0)
  
  metaD <- mat %>% dplyr::select(ct_site, disturbance)
  counts <- as.matrix(mat %>% dplyr::select(-ct_site, -disturbance))
  
  # mean NP community (centroid)
  NP_idx <- which(metaD$disturbance == "Low")
  NP_centroid <- colMeans(counts[NP_idx, , drop = FALSE])
  
  # for each non-NP site, calculate balanced bray-curtis similarity to NP centroid
  results <- map_dfr(seq_len(nrow(counts)), function(i) {
    pair <- rbind(NP_centroid, counts[i, ])
    rownames(pair) <- c("NP_centroid", metaD$ct_site[i])
    
    if (any(rowSums(pair) == 0)) {
      sim <- NA
    } else {
      sim <- 1 - bray.part(pair)$bray.bal[1]
    }
    
    data.frame(ct_site = metaD$ct_site[i],
               disturbance = metaD$disturbance[i],
               comp_sim = sim)
  })
}

cs_species <- comp_similarity(meta[!(meta$species %in% domes),], "species")
cs_cluster <- comp_similarity(clustID, "cluster_label")

# ── 3. COMBINE INTO BII ───────────────────────────────────────────────────────
# BII = abundance component × compositional similarity component
# NP gets BII = 1 by definition

calc_BII <- function(ab, cs, label) {
  ab %>%
    left_join(cs, by = c("ct_site", "disturbance")) %>%
    mutate(
      BII = abundance_BII * comp_sim,
      metric = label
    ) %>%
    dplyr::select(ct_site, disturbance, abundance_BII, comp_sim,
                  BII, metric)
}

BII_species <- calc_BII(ab_species, cs_species, "Species")
BII_cluster <- calc_BII(ab_cluster, cs_cluster, "VOTU")

BII_all <- bind_rows(BII_species, BII_cluster)
print(BII_all)

BII_all$disturbance = factor(BII_all$disturbance, levels = c("Low", "Medium", "High"))

library(Hmisc)

BII_all <- BII_all %>%
  group_by(metric) %>%
  mutate(BII = BII / mean(BII[disturbance == "Low"], na.rm = TRUE))

ggplot(BII_all, aes(x = disturbance, y = BII, colour = metric)) +
  # geom_point(data = BII_all, aes(x = disturbance, y = BII), position = position_dodge(width = 0.4), alpha = 0.4) +
  stat_summary(fun.data = mean_cl_normal, geom = "pointrange",
               position = position_dodge(width = 0.4)) +
  labs(x = "Disturbance level", y = "Biodiversity Intactness Index (BII)", colour = "") +
  theme_classic() +
  scale_colour_manual(values = c("Species" = "#2166ac", 
                                 "VOTU" = "#d6604d")) +
  theme(text = element_text(size = 12)) +
  ggtitle("c)") +
  geom_hline(yintercept = 1, linetype = "dashed", colour = "grey50")

# 3. what are the clusters made up of?
# have a look at cluster 'purity' - are they actually linked to species, functional groups, time zones 
# or something else?


# stacked bar plot - for every cluster label, what proportion of images are made up of each species?


clusta = ggplot(clustID, aes(x = as.factor(cluster_label), fill = species)) +
  geom_bar(position = "fill") +
  labs(x = "VOTU Label", y = "Proportion of Images", fill = "Species") +
  theme_classic() +
  theme(text = element_text(size = 12)) +
  ggtitle("a)")
# Label 0 = 60% bird + peacock +jungle fowl total 80%
#   Label 1 = elephant + rhino
# Label 2 = 100% rhino
# LLabel 3 = chital + other deer
# 4 = chital + wild boar
# 5 = chital
# 6 = macaque
# 7 = barking deer + sambar
# 8 = chital
# 9 = grey langur
# 10 = grey langur
# 11 = chital 
# 12 = chital

clustID$datetime = as.POSIXct(clustID$datetime, format = "%Y-%m-%d %H:%M:%S")
clustID$hour = hour(clustID$datetime)

clustb = ggplot(clustID, aes(x = as.factor(cluster_label), fill = as.factor(hour))) +
  geom_bar(position = "fill") +
  labs(x = "VOTU label", y = "Proportion of Images", fill = "Time of day (24h)") +
  theme_classic() +
  theme(text = element_text(size = 12)) +
  ggtitle("b)")

clusta+clustb & plot_layout(nrow = 2)

# make this into a table format

clust_freq = clustID %>%
  group_by(cluster_label, species) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(cluster_label) %>%
  mutate(prop = n / sum(n)) %>%
  arrange(cluster_label, desc(prop))

