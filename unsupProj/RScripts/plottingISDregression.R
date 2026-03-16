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

cbs = read.csv("results/clustercountsbysite.csv")
countbs = read.csv("results/countsbysite.csv")
reg = read.csv("results/regression_summary.csv")

# the actual list of images that went into this
meta = read.csv("data/nepal_cropsmeta_PB.csv") %>%
  filter(species != "human") %>%
  filter(species != "vehicle") %>%
  filter(SetID == 'train')
unique(meta$species)
# what is the date range of this dataset?
range(meta$date)
# there is one row that doesn't seem to have a date - remove this
meta$datetime[meta$date==''] = '2019-03-25 00:08:37'
meta$date[meta$date==''] = '2019-03-25'
meta$time[meta$time==''] = '00:08:37'
meta$time_hour[meta$time_hour==''] = 0
meta$time_min[meta$time_min==''] = 8
meta$time_sec[meta$time_sec==''] = 37

range(meta$date)
# the range of the training data is from 2nd Feb to 25th May
class(meta$date)
meta$date = as.Date(meta$date)
hist(meta$date, breaks = "weeks")
# for now we will keep in all the data, but worth noting that the ferreira paper 
# used cutoff points of 15th March - 15th April. 
# plot regression results 

ggplot(cbs, aes(x = distinct_label_count, y = distinct_species_count, colour = as.factor(n_components),
                linetype = hdbscan_cfg)) +
  # geom_point(alpha = 0.1) +
  geom_smooth(method = "lm", se = FALSE) +
  facet_wrap(~model) +
  theme_bw() +
  ylim(0, 20) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey") +
  labs(x = "ImageScape Diversity", y = "Species Richness", colour = "UMAP Dimensions", linetype = "HDBSCAN Config")


ggplot(cbs[cbs$hdbscan_cfg=='leaf_2pct',], aes(x = distinct_label_count, y = distinct_species_count, colour = as.factor(n_components)
                )) +
  geom_point(alpha = 0.2) +
  geom_smooth(method = "lm", se = FALSE) +
  facet_wrap(~model) +
  ylim(0, 20) +
  theme_bw() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey") +
  labs(x = "ImageScape Diversity", y = "Species Richness", colour = "UMAP Dimensions", linetype = "HDBSCAN Config")


ggplot(cbs[cbs$hdbscan_cfg=='leaf_0p5pct',], aes(x = distinct_label_count, y = distinct_species_count, colour = as.factor(n_components)
)) +
  geom_point(alpha = 0.2) +
  geom_smooth(method = "lm", se = FALSE) +
  facet_wrap(~model) +
  ylim(0, 20) +
  theme_bw() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey") +
  labs(x = "ImageScape Diversity", y = "Species Richness", colour = "UMAP Dimensions", linetype = "HDBSCAN Config")


ggplot(cbs[cbs$hdbscan_cfg=='leaf_1pct',], aes(x = distinct_label_count, y = distinct_species_count, colour = as.factor(n_components)
)) +
  geom_point(alpha = 0.2) +
  geom_smooth(method = "lm", se = FALSE) +
  facet_wrap(~model) +
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
reg$wild = ifelse(grepl("wild", reg$model), "wild", "all")
reg$model_name = gsub("_wild", "", reg$model)
# make a column for r2 all and r2 wild
reg_wide = pivot_wider(reg, 
                       id_cols = c("model_name", "n_components","min_cluster_size_pc"), 
                       names_from = wild,
                       values_from = c('correlation_r', 'noise_ratio_global'))
# select the highest r value for each model
bold_rows_all <- reg_wide %>%
  mutate(row_num = row_number()) %>%
  group_by(model_name) %>%
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
  set_header_labels(model_name = "Model", n_components = "UMAP Dimensions",
                    min_cluster_size_pc = "Min Cluster Size (%)",
                    correlation_r_all = "All images", 
                    correlation_r_wild = "Wild images only",
                    noise_ratio_global_all = 'All images',
                    noise_ratio_global_wild = 'Wild images only') %>%
  set_table_properties(layout = "autofit", width = .8)%>%
  hline(i = c(12, 24, 36), border = fp_border_default()) %>%
  bold(i = bold_rows_all, j = ~ correlation_r_all) %>%
  bold(i = bold_rows_wild, j = ~ correlation_r_wild) %>%
  add_header_row(values = c("", "R2", "Noise Ratio"), colwidths = c(3, 2,2))
f1

# save f1 as png
save_as_image(f1, "results/figures/regsum_full.png", width = 8, height = 4, units = "in", res = 300)


# alternatively, lets choose the top 5 performing models and present these
reg_arr = arrange(reg_wide, desc(correlation_r_all))
f2_all  = flextable(reg_arr[1:5,]) %>%
  set_header_labels(model_name = "Model", n_components = "UMAP Dimensions",
                     min_cluster_size_pc = "Min Cluster Size (%)",
                     correlation_r_all = "All images", 
                     correlation_r_wild = "Wild images only",
                     noise_ratio_global_all = 'All images',
                     noise_ratio_global_wild = 'Wild images only') %>%
  add_header_row(values = c("", "R2", "Noise Ratio"), colwidths = c(3, 2,2)) %>%
  set_table_properties(layout = "autofit", width = .8)
f2_all
save_as_image(f2_all, "results/figures/regsum_allimgs_top5.png", width = 8, height = 4, units = "in", res = 300)

reg_arr = arrange(reg_wide, desc(correlation_r_wild))
f2_wild  = flextable(reg_arr[1:5,]) %>%
  set_header_labels(model_name = "Model", n_components = "UMAP Dimensions",
                    min_cluster_size_pc = "Min Cluster Size (%)",
                    correlation_r_all = "All images", 
                    correlation_r_wild = "Wild images only",
                    noise_ratio_global_all = 'All images',
                    noise_ratio_global_wild = 'Wild images only') %>%
  add_header_row(values = c("", "R2", "Noise Ratio"), colwidths = c(3, 2,2)) %>%
  set_table_properties(layout = "autofit", width = .8)
f2_wild
save_as_image(f2_wild, "results/figures/regsum_wildimgs_top5.png", width = 8, height = 4, units = "in", res = 300)


n_distinct(meta$species)
`# ANALYSIS PLAN

# Data prep 
# connect dataset with environmental variables of all the camera trap sites
# also need a dataset with each embedding vector connected to the image name, 
# so that time stamp can be extracted.

siteinfo = read.csv("data/siteVariable_149sites_Nepal_Oct2020.csv")
cbs1 = left_join(cbs, siteinfo, by = c("ct_site" = "CT_site"))

# need to get a measure of survey effort - number of camera trap days for each CT site.
se = read.csv("data/SurveyEffort_Mar-Apr_2019_corrected.csv")
range(meta$datetime)
range(se$X)

# we cannot get full number of camera trap days because my survey effort table is
# only between 15th March - 15th April.





# 1. ISD AS AN ECOSYSTEM CONDITION INDEX
# used glms or gam to predict ISD over habitat gradient. compare coefficients with 
# model from species richness
# will need to include survey effort.

# first, lets plot species richness across land-use types

# subset to one model type - the once that has highest performance in f2
# convnextL wild with leaf_1% and 8 umap dimensions
# this has a correlation of 0.74 & noise ratio of 0.53
topcbs = cbs1 %>%
  filter(model == "convnextL_wild") %>%
  filter(n_components == 8) %>%
  filter(hdbscan_cfg == "leaf_1pct")

topcbs$Management = factor(topcbs$Management, levels = c("NP", "BZ", "OBZ"))

a = ggplot(data = topcbs, aes(x = distinct_label_count, y = distinct_species_count)) +
  geom_point() +
  geom_smooth(method = "lm") +
  theme_bw() +
  labs(x = "ImageScape Diversity", y = "Species Richness") +
  annotate("text", x = 10, y = 14, label = "R2 = 0.74", size = 6) +
  
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey") +
  ggtitle("a)") +
  ylim(0, 15) +
  theme(text = element_text(size = 16))

# species richness & ISD over management zone
b = ggplot(data = topcbs, aes(x = Management, y = distinct_species_count)) +
  geom_boxplot() +
  theme_bw() +
  labs(x = "Management Zone", y = "Species Richness") +
  ggtitle("b)") +
  theme(text = element_text(size = 16))
b
c= ggplot(data = topcbs, aes(x = Management, y = distinct_label_count)) +
  geom_boxplot() +
  theme_bw() +
  labs(x = "Management Zone", y = "ISD") +
  ggtitle("c)") +
  theme(text = element_text(size = 16))
c
a+b+c

# species richness over forest cover

d = ggplot(data = topcbs, aes(x = propForest500, y = distinct_species_count)) +
  geom_point() +
  geom_smooth(method = "lm") +
  theme_bw() +
  labs(x = "Proportion Forest Cover (500m radius)", y = "Species Richness") +
  ggtitle("d)") +
  theme(text = element_text(size = 16))

e = ggplot(data = topcbs, aes(x = propForest500, y = distinct_label_count)) +
  geom_point() +
  geom_smooth(method = "lm") +
  theme_bw() +
  labs(x = "Proportion Forest Cover (500m radius)", y = "ISD") +
  ggtitle("e)") +
  theme(text = element_text(size = 16))


a+plot_spacer()+b+c+d+e +plot_layout(ncol = 2)


# originally i correlated ISD with species richness
# total number of images per camera trap site
domes = c('buffalo', 'cow', 'dog', 'domestic_cat', 'domestic_chicken', 'domestic elephant', 'goat', 'sheep')

meta_count = meta %>%
  filter(!(species %in% domes)) %>%
  group_by(ct_site) %>%
  summarise(image_count = n())

topcbs = left_join(topcbs, meta_count, by = c("ct_site" = "ct_site"))
# there is a correlation between number of images and species richness/ISD, 
# which is expected because more images = more chances to capture different species.

plot(log(topcbs$image_count), topcbs$distinct_label_count)
plot(log(topcbs$image_count), topcbs$distinct_species_count)

# 
# linear model of species richness over management zone and forest cover

# Linear modelling of ISD -------------------------------------------------

# terms all need to be scaled so they match - species richness, forest cover and image count

topcbs = topcbs %>% mutate(
  distinct_species_count_scaled = as.numeric(scale(distinct_species_count)),
  propForest500_scaled = as.numeric(scale(propForest500)))

# we should model with poisson distribution, but lets check for overdispersion
m1 = glm(distinct_species_count ~ Management*propForest500_scaled + 
           offset(log(image_count)), 
         data = topcbs, family = "poisson")
summary(m1)

m2 = glm(distinct_label_count ~ Management*propForest500_scaled + 
           offset(log(image_count)), 
         data = topcbs, family = "poisson")
summary(m2)


check_overdispersion(m1)
check_overdispersion(m2)

# both of these models are overdispersed, 
# so we should use a negative binomial distribution instead

library(MASS)

m3_sr = glm.nb(distinct_species_count ~ Management*propForest500_scaled + 
           offset(image_count), 
         data = topcbs)
summary(m3_sr)
m4_isd = glm.nb(distinct_label_count ~ Management*propForest500_scaled + 
           offset(log(image_count)), 
         data = topcbs)
summary(m4_isd)
# there is potentially a non linear relationship between forest cover 
# and species richness/ISD, so we could also try a GAM 
# with a negative binomial distribution

library(mgcv)

m_gamsr_s <- gam(distinct_species_count ~ Management + s(propForest500_scaled) + 
               offset(log(image_count)), data = topcbs, family = nb(), method = 'REML')
summary(m_gamsr_s)

m_gam<- gam(distinct_label_count ~ Management + s(propForest500_scaled) + 
               offset(log(image_count)), data = topcbs, family = nb())
summary(m_gam)

plot(m_gamsr_s, shade = TRUE, pages = 1)

plot(m_gam, shade = TRUE, pages = 1)


sr_coefs <- tidy(m3_sr, conf.int = TRUE, exponentiate = TRUE) %>% 
  mutate(response = "Species Richness")
isd_coefs <- tidy(m4_isd, conf.int = TRUE, exponentiate = TRUE) %>% 
  mutate(response = "ISD")

bind_rows(sr_coefs, isd_coefs) %>%
  filter(!term %in% c("(Intercept)")) %>%
  mutate(term = recode(term,
                       "ManagementBZ" = "Buffer Zone (BZ)",
                       "ManagementOBZ" = "Outside Buffer Zone (OBZ)",
                       "propForest500_scaled" = "Forest Cover (500m)",
                       "ManagementOBZ:propForest500_scaled" = "OBZ:Forest Cover",
                       "ManagementBZ:propForest500_scaled" = "BZ:Forest Cover")) %>%
  ggplot(aes(x = estimate, y = term, colour = response,
             xmin = conf.low, xmax = conf.high)) +
  geom_pointrange(position = position_dodge(width = 0.4)) +
  geom_vline(xintercept = 1, linetype = "dashed", colour = "grey50") +
  scale_colour_manual(values = c("Species Richness" = "#2166ac", "ISD" = "#d6604d")) +
  labs(x = "Rate Ratio", y = NULL, colour = NULL) +
  theme_classic()

library(ggeffects)

pred_mgmt <- ggpredict(m4_isd, terms = "Management", condition = c(image_count = mean(topcbs$image_count)))

ggplot() +
  geom_jitter(data = topcbs, aes(x = Management, y = distinct_label_count),
              width = 0.15, alpha = 0.4) +
  geom_pointrange(data = pred_mgmt, 
                  aes(x = x, y = predicted, ymin = conf.low, ymax = conf.high),
                  colour = "#d6604d", size = 1) +
  labs(x = "Management Zone", y = "ISD") +
  theme_classic()

# there is an issue where image count is changing the effect of species richness. 
ggplot(topcbs, aes(x = Management, y = distinct_label_count)) +
  geom_jitter(aes(size = image_count, colour = image_count), width = 0.15, alpha = 0.6) +
  scale_colour_viridis_c(option = "magma") +
  labs(x = "Management Zone", y = "ISD", size = "Image Count", colour = "Image Count") +
  theme_classic()

ggplot(topcbs, aes(x = Management, y = image_count)) +
  geom_boxplot(outlier.shape = NA) +
  geom_jitter(width = 0.15, alpha = 0.5) +
  labs(x = "Management Zone", y = "Image Count") +
  theme_classic()

# image count is correlated with management zone. this is actually inapppropriate to use as an offset, 
# because it is not a measure of survey effort, but rather a result of the underlying ecological conditions (more animals in NP = more images & more species)

# i need to measure survey effort properly. 
# realistically i need to only use images from the survey period as the images from random sampling are not systematically sampled, 
# so will not be related to survey effort. 
# this needs to happen in the python script because the data i have is already summarised.
# for now, just use the survey effort i have

camtrapdays = se[,-1] %>%
  colSums(na.rm = TRUE) %>%
  data.frame(ct_site = names(.), camtrapdays = .)

topcbs = topcbs %>%
  left_join(camtrapdays, by = "ct_site")


# limit topCBS to 15th March - 15th April. 

m5_sr = glm(distinct_species_count ~ Management*propForest500_scaled - 1 + 
                 offset(log(camtrapdays)), 
               data = topcbs, family = "poisson")
check_overdispersion(m5_sr)
# no overdispersion here, so we can use poisson distribution.
summary(m5_sr)

m5_isd = glm(distinct_label_count ~ Management*propForest500_scaled -1 +
                 offset(log(camtrapdays)), 
               data = topcbs, family = "poisson")
summary(m5_isd)
check_overdispersion(m5_isd)
# overdispersion detected, but it just on the limit
m5_isdnb = glm.nb(distinct_label_count ~ Management + propForest500_scaled + 
               offset(log(camtrapdays)), 
             data = topcbs)
summary(m5_isdnb)
# nb can't converge
# we can try quasipoisson instead
m5_isdqp = glm(distinct_label_count ~ Management * propForest500_scaled - 1  + 
               offset(log(camtrapdays)), 
             data = topcbs, family = quasipoisson())
summary(m5_isdqp)
# this does not give an AIC. It is possible to compare coefficients with the poisson model, 
# but bare in mind that the standard error will be wider for the quasipoisson model, so the confidence intervals will be wider and there may be less significant results.

# compare model coefficients
sr_coefs <- tidy(m5_sr, conf.int = TRUE, exponentiate = TRUE) %>% 
  mutate(response = "Species Richness")
isd_coefs <- tidy(m5_isdqp, conf.int = TRUE, exponentiate = TRUE) %>% 
  mutate(response = "ISD")

df1 = bind_rows(sr_coefs, isd_coefs) %>%
  mutate(term = recode(term,
                       "ManagementNP" = "National Park (NP)",
                       "ManagementBZ" = "Buffer Zone (BZ)",
                       "ManagementOBZ" = "Outside Buffer Zone (OBZ)",
                       "propForest500_scaled" = "Forest Cover (500m)",
                       "ManagementOBZ:propForest500_scaled" = "OBZ:Forest Cover",
                       "ManagementBZ:propForest500_scaled" = "BZ:Forest Cover"))
df1$term = factor(df1$term,levels = c("National Park (NP)",
                                  "Buffer Zone (BZ)",
                                  "Outside Buffer Zone (OBZ)",
                                  "Forest Cover (500m)",
                                  "BZ:Forest Cover",
                                  "OBZ:Forest Cover"))

coefp = ggplot(df1, aes(x = estimate, y = term, colour = response,
             xmin = conf.low, xmax = conf.high)) +
  geom_pointrange(position = position_dodge(width = 0.4)) +
  geom_vline(xintercept = 1, linetype = "dashed", colour = "grey50") +
  scale_colour_manual(values = c("Species Richness" = "#2166ac", "ISD" = "#d6604d")) +
  labs(x = "Rate Ratio", y = NULL, colour = NULL) +
  theme_classic() +
  theme(text = element_text(size = 16))
coefp
library(ggeffects)

pred_mgmt <- ggpredict(m5_isdqp, terms = "Management", condition = c(camtrapdays = mean(topcbs$camtrapdays)))

mgmtisd = ggplot() +
  geom_jitter(data = topcbs, aes(x = Management, y = distinct_label_count),
              width = 0.15, alpha = 0.4) +
  geom_pointrange(data = pred_mgmt, 
                  aes(x = x, y = predicted, ymin = conf.low, ymax = conf.high),
                  colour = "#d6604d", size = 1) +
  labs(x = "Management Zone", y = "ImageScape Diversity") +
  theme_classic() +
  theme(text = element_text(size = 16)) +
  ggtitle("b)")

pred_mgmt_sr<- ggpredict(m5_sr, terms = "Management", condition = c(camtrapdays = mean(topcbs$camtrapdays)))

mgmtsr = ggplot() +
  geom_jitter(data = topcbs, aes(x = Management, y = distinct_species_count),
              width = 0.15, alpha = 0.4) +
  geom_pointrange(data = pred_mgmt_sr, 
                  aes(x = x, y = predicted, ymin = conf.low, ymax = conf.high),
                  colour = "#d6604d", size = 1) +
  labs(x = "Management Zone", y = "Species Richness") +
  theme_classic() +
  theme(text = element_text(size = 16)) +
  ggtitle("a)")

mgmtsr + mgmtisd

# 2. ISD for measuring change in community
# compute pairwise dissimilarity across sites (NP vs BZ vs OBZ) and measure 
# community change across the pressure gradient. compare with actual species turnover

# 3. what are the clusters made up of?
# have a look at cluster 'purity' - are they actually linked to species, functional groups, time zones 
# or something else?




