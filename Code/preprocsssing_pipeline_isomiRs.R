library(plotrix)
library(Hmisc)
library(FSA)
library(reshape2)
library(DESeq2)
library(sva)
library(edgeR)
library(gamlss)
library(readxl)
library(rMIDAS)
library(survival)
library(survMisc)
library(OptimalCutpoints)
library(maxstat)
library(rolr)
library(ggplot2)
library(GGally)
library(plotly)
library(memisc)
library(rms)
library(pec)
library(dplyr)
library(glmnet)
library(bootStepAIC)
library (forestmodel)
library(viridis)
library(ggrepel)
library(gridExtra)
library(factoextra)
library(NbClust)
library(psych)  
library(cowplot)
library(egg)
library(table1)
library(grid)
library(ggpubr)
library(ggbreak)
library(questionr)
library(RVAideMemoire)
library(stringr)
#library(dbscan)
library(DescTools)
library(rcompanion)
library(survivalAnalysis)
#library(enrichR)
#library(STRINGdb)
#library(biomaRt)
library(survminer)
library(writexl)
setwd("D:/yahel/phd/isomiRs")

############################################
#     IsoMIRs table of contents            #
############################################

# 5' changes are marked in suffix  - "s" / else the changes are on the 3'
# Deletions are lowercase  / Insertions are upppercase
# exmaple
# ACGT (ref)
#  CGT .as
# ACG  .t
# ACGTT .T
#AACGT .As
# Mutations are indicated with reference symbol, position and new symbol.
# Consecutive mutations will not be merged into MNVs.
# The position is relative to the reference, 
# so preceding (5') indels will not offset it
# exmaple
# ACGT (ref)
# ATGT .C2T
# ATAT .C2T.G3A
#AATGT .As.C2T (not C3T)
# TGT .as.C2T (not C1T)


############################################
#       Required built functions           #
############################################
## a function that merges 2 DFs to one ##

# df1 / df2 - dataframe exported from CLC workbench "extract isoMIRs" tool
# column.name - name of the second column to be merged
# amb.filter - Boolean, if T, filters out ambiguous reads isoMIRs - with 2 or more different matches of the sequence
# canonical - Boolean, if T, filters out all the non-canonical isoMIRs
# combine.all - Boolean, if F, filters out all isoMIRs not shared across df1 & df2
# if T, merge df1 & df2, non shared isoMIRs will remain and the value 0 will be added to the chamber in the df they don't exist in

combine.df <- function(df1, df2, column.name,
                       amb.filter = F, canonical = F, combine.all = F){
  if (amb.filter) {
    if (!is.null(df1$Ambiguous)) {
      df1 <- df1[df1$Ambiguous == 'false',]
    }
    df2 <- df2[df2$Ambiguous == 'false',]
  }
  if (canonical) {
    df1 <- df1[sapply(strsplit(df1$Name,'\\.'), length) == 1,]
    df2 <- df2[sapply(strsplit(df2$Name,'\\.'), length) == 1,]
  }
  if (isFALSE(combine.all)) {
    if (!is.null(df1$Sequence) | !is.null(df1$Ambiguous)) {
      df1 <- df1[,c(1,3)]
    }
    df2 <- df2[,c(1,3)]
    shared_mat <- cbind.data.frame(df1[df1$Name %in% df2$Name,],df2$Count[df2$Name %in% df1$Name])
    colnames(shared_mat)[ncol(shared_mat)] <- column.name
    return(shared_mat)
  } else{
    if (!is.null(df1$Sequence) | !is.null(df1$Ambiguous)) {
      df1 <- df1[,c(1,3)]
    }
    df2 <- df2[,c(1,3)]
    shared_mat <- merge(x = df1,y = df2,by = 'Name',all = T)
    colnames(shared_mat)[ncol(shared_mat)] <- column.name
    return(shared_mat)
  }
}

############################################
#   Load raw isomiRs counts and combine    #
############################################

shared_mat <- NULL
for (i in dir(path = 'D:/yahel/phd/isomiRs/CLC/excel_results/isoMIRS_ALS_CTL/',pattern = '.xlsx')) {
  if (i == "~$BLT00086 (isomiR counts).xlsx") {
    next
  }
  path <- paste0('D:/yahel/phd/isomiRs/CLC/excel_results/isoMIRS_ALS_CTL/',i)
  mat <- read_excel(path = path)
  add.name <- strsplit(i,' ')[[1]][1]
  if (is.null(shared_mat)) {
    shared_mat <- mat
    colnames(shared_mat)[3] <- add.name
  }else{
    shared_mat <- combine.df(shared_mat,mat,add.name,canonical = F,combine.all = T)
    print(add.name)
  }
}


##############################################
#          Filter all isomiRs data           #
##############################################

all_isoMIRs <- shared_mat

rownames(all_isoMIRs) <- all_isoMIRs$Name ; all_isoMIRs <- all_isoMIRs[,-1]
colnames(all_isoMIRs) <- gsub(pattern = ' ',replacement = '',x = colnames(all_isoMIRs))
colnames(all_isoMIRs) <- gsub(pattern = '_',replacement = '',x = colnames(all_isoMIRs))

all_isoMIRs <- all_isoMIRs[,colnames(all_isoMIRs) %in% clinical$`index case`]

min.count <- 5 # define the minimum average count per isomiR per subject 
iso_sum <- apply(all_isoMIRs, 1, function(x)(sum(as.numeric(x),na.rm = T)))
filter.umi <- iso_sum >= ncol(all_isoMIRs)*min.count

filtered_umi_mat <- all_isoMIRs[filter.umi,]
filtered_umi_mat <- as.matrix(filtered_umi_mat)

dim(filtered_umi_mat)
# plot bar plots for n-isomiRs per cutoff #

peaks <- NULL
for (cutoff in seq(0,100,1)) {
  peaks <- append(peaks,sum(iso_sum >= ncol(all_isoMIRs)*cutoff))
}

peak_df <- cbind.data.frame('Cutoff' = seq(0,100,1), 'n_isomiRs' = peaks)

peak_bars <- ggplot(peak_df,aes(x = Cutoff, y = n_isomiRs)) + geom_col(width = 0.5) + labs(y = 'Number of isomiRs included', x = 'Cutoff (UMI)') +
  theme(panel.background = element_blank(),axis.line = element_line(color = 'lightgrey'),legend.position = 'top',legend.title = element_blank(),
        legend.text = element_text(size = 18),axis.title = element_text(size = 26,face = 'bold',hjust = 0.5),
        axis.text =  element_text(size = 20,face = 'bold',color = 'black'),axis.text.x =  element_text(angle = 45,hjust = 1)) +
  scale_y_break(c(6000,28000), scale='free') + scale_x_continuous(breaks = seq(0,100,5)) + 
  geom_segment(aes(x = 10, y = 3000, xend = 5, yend = 2100),arrow = arrow(length = unit(.5, "cm")),size = 2, color = 'red') + 
  annotate(geom = 'text',x = 35, y = 3300,label = 'Chosen noise cutoff',fontface = "bold",size = 8, color = 'red')


# names and number of canonical forms #
levels(as.factor(unlist(sapply(rownames(filtered_umi_mat), function(x)(strsplit(x,'\\.')[[1]][1])))))

# number of isomiRs per canonical form #
table(unlist(sapply(rownames(filtered_umi_mat), function(x)(strsplit(x,'\\.')[[1]][1]))))
##############################################
#          Run multilevel imputation         #
##############################################

# read clinical data #
clinical <- read_excel(path = 'D:/yahel/phd/isomiRs/Clinical info for Yahel.xls',sheet = 'Merged')

clinical$`index case` <- gsub(pattern = ' ',replacement = '',x = clinical$`index case`)
clinical$`index case` <- gsub(pattern = '_',replacement = '',x = clinical$`index case`)

clinical <- clinical[clinical$`index case` %in% final_uk_p,]

# filter for existing subjects in the isomir data set
iso_data <- cbind.data.frame('index case' = colnames(filtered_umi_mat),t(filtered_umi_mat))

iso_data$`index case` <- gsub(pattern = ' ',replacement = '',x = iso_data$`index case`)
iso_data$`index case` <- gsub(pattern = '_',replacement = '',x = iso_data$`index case`)

# merge all uni counts and clinical features to one df #
merged_data <- merge(iso_data,clinical,by = 'index case', all = F)
rownames(merged_data) <- rownames(iso_data) ; merged_data <- merged_data[,-c(1,2084,2086,2087,2090,2093,2094,2096)]

# define or scale the type of variables for algorithm training #
binary <- c('sex','onset','treatment','outcome')
converted_vars <- convert(data = merged_data,bin_cols = binary,cat_cols = c(),minmax_scale = T)

# Train the algorithm #
train_model <- train(data = converted_vars,training_epochs = 40,layer_structure = c(512,512,512),
                     input_drop = 0.95,seed = 42,learn_rate = 0.000001,
                     vae_layer = T)

# generate 50 different imputations per NA value #
# if there aren't category variables than cat_coalesce = F #
m <- 50
imputaions_sims <- complete(train_model,m = m,cat_coalesce = F)

# average all imputaions across m repeats #
sum.mat <- NULL
for (matrix in 1:length(imputaions_sims)) {
  if (is.null(sum.mat)) {
    sum.mat <- imputaions_sims[[matrix]][,1:(ncol(merged_data) - 9)]
  }else{
    sum.mat <- sum.mat + imputaions_sims[[matrix]][,1:(ncol(merged_data) - 9)]
  }
}

imputated_umi <- sum.mat / m

rownames(imputated_umi) <- rownames(merged_data)
imputated_umi <- t(imputated_umi)


##########################################
#    Batch correct all isoMIRs data      #
##########################################

# ----------------------------------------------- #
#               For ALS patients data             #
# ----------------------------------------------- #

batch_path <- '/home/labs/hornsteinlab/yahelc/R/isoMIRs/dats/All_samples_enrolment.txt'

batch_info <- as.data.frame(t(read.delim(batch_path,nrows = 5)))
colnames(batch_info) <- batch_info[1,] ; batch_info <- batch_info[-1,]

batch_info$sample <- gsub(pattern = ' ',replacement = '',x = batch_info$sample)
batch_info$sample <- gsub(pattern = '_',replacement = '',x = batch_info$sample)
batch_info$sex <- gsub(pattern = ' ',replacement = '',x = batch_info$sex)

batch_info <- batch_info[batch_info$sample %in% colnames(imputated_umi),]

# comBat needs normalized counts so I use cpm #
#when working on data after stability filtration change "norm_counts_all_isoMIRs" -> "stable_imputated"

cpmdata <- cpm(imputated_umi,log=TRUE, normalized.lib.sizes=TRUE)

comdata <- ComBat(dat = cpmdata,batch = clinical$Batch[match(x = clinical$`index case`,colnames(imputated_umi))])

# I reversed the log transformation by cpm in order to correct by DESeq corrrection #

revertdata = t(((2^t(comdata))*colSums(imputated_umi)/1000000))

#########################################################
#     correct imputed UMI data by geometrical mean      #
#########################################################

geom_mean <- apply(revertdata, 1, geometric.mean)
DESeq_norm <- revertdata / geom_mean
correct_factor <- apply(DESeq_norm, 2, median)
norm_counts_all_isoMIRs <- t(t(revertdata) / correct_factor)

colnames(norm_counts_all_isoMIRs) <- gsub(pattern = '_',replacement = '',x = colnames(norm_counts_all_isoMIRs))



# ------------------------------------ #
#    replace imputated data with 0     #
# ------------------------------------ #

dim(norm_counts_all_isoMIRs) 

colnames(norm_counts_all_isoMIRs)
norm_counts_all_isoMIRs[is.na(filtered_umi_mat)] <- 0

## Merge dataset with clinical data ##

corrected_umi <- t(rbind.data.frame(norm_counts_all_isoMIRs,'outcome' = clinical$outcome[match(table = colnames(norm_counts_all_isoMIRs),x = clinical$`index case`)],
                                    'Survival.from.onset' = clinical$`Survival from onset`[match(table = colnames(norm_counts_all_isoMIRs),x = clinical$`index case`)],
                                    'Survival.from.enrolment' = clinical$`Survival from enrolment`[match(table = colnames(norm_counts_all_isoMIRs),x = clinical$`index case`)]))




###############################################
#           FOR REPLICATION COHORT            #
###############################################

#########################################################
#          Filter all isomiRs data repliction           #
#########################################################

all_isoMIRs_replication <- read.csv('data_matrix/create_plasma_isomiR_matrix.csv')
rownames(all_isoMIRs_replication) <- all_isoMIRs_replication$Name ; all_isoMIRs_replication <- all_isoMIRs_replication[,-c(1:2)]

# change the column names to fit clinical data matrix #
colnames(all_isoMIRs_replication) <- gsub(x = colnames(all_isoMIRs_replication),pattern = 'X',replacement = 'BAC_')
colnames(all_isoMIRs_replication) <- gsub(x = colnames(all_isoMIRs_replication),pattern = 'Bac',replacement = 'BAC')
colnames(all_isoMIRs_replication) <- sapply(colnames(all_isoMIRs_replication), function(x)(paste(strsplit(x,'_')[[1]][-4],collapse = '_')))

# read clinical data #
clinical_replication <- read_excel(path = 'data_matrix/Hornstein PGB Clin Data - plasma & CSF (2023.09.05).xlsx',sheet = 'Plasma clin data')

# filter only unique plasma samples
clinical_replication <- clinical_replication[clinical_replication$CollNum_p ==1,]

# change hot-one-key for onset to one column #
clinical_replication$Riluzole <- c('No', 'Yes')[c(clinical_replication$Riluzole ==1)+1] 
for (col in 7:9) {
  if (col == 7) {
    clinical_replication[,col] <- c('No','Bulbar')[c(clinical_replication[,col] ==1)+1]
  } else if(col == 8){
    clinical_replication[,col] <- c('No','Limb')[c(clinical_replication[,col] ==1)+1]
  } else{
    clinical_replication[,col] <- c('No','other')[c(clinical_replication[,col] ==1)+1]
  }
}

clinical_replication$disease_onset <- apply(clinical_replication[,7:9],1,function(x)(paste(x,collapse = '_')))
clinical_replication$disease_onset <- sapply(clinical_replication$disease_onset, function(x)(str_remove_all(string = x, 'No_|_No')))


# filter out for unique plasma samples that were

all_isoMIRs_replication <- all_isoMIRs_replication[,colnames(all_isoMIRs_replication) %in% clinical_replication$ID]


min.count <- 5 # define the minimum average count per isomiR per subject 
iso_sum <- apply(all_isoMIRs_replication, 1, function(x)(sum(as.numeric(x),na.rm = T)))
filter.umi <- iso_sum >= ncol(all_isoMIRs_replication)*min.count

filtered_umi_mat_replication <- all_isoMIRs_replication[filter.umi,]
filtered_umi_mat_replication[filtered_umi_mat_replication == 0] <- NA

filtered_umi_mat_replication <- as.matrix(filtered_umi_mat_replication) # 1999 isomiRs

# names and number of canonical forms #
levels(as.factor(unlist(sapply(rownames(filtered_umi_mat_replication), function(x)(strsplit(x,'\\.')[[1]][1])))))

# number of isomiRs per canonical form #
table(unlist(sapply(rownames(filtered_umi_mat_replication), function(x)(strsplit(x,'\\.')[[1]][1]))))
##############################################
#          Run multilevel imputation         #
##############################################

# filter for existing subjects in the isomir data set
iso_data <- cbind.data.frame('ID' = colnames(filtered_umi_mat_replication),t(filtered_umi_mat_replication))

# merge all uni counts and clinical features to one df #
merged_data <- merge(iso_data,clinical_replication,by = 'ID', all = F)
rownames(merged_data) <- merged_data[,1]  ; merged_data <- merged_data[,-1]
merged_data <- merged_data[,-match(table = colnames(merged_data),x = c('PID','CollNum_p','SampleID_p','Os3_Bulbar_dm',
                                                                       'Os3_Limb_dm','Os3_Other_dm','deltaFRS','Endpt_reason'))]


# define or scale the type of variables for algorithm training #
binary <- c('Sex_c','Endpt_LV','Riluzole')
cat_cols <- c('disease_onset')
converted_vars <- convert(data = merged_data,bin_cols = binary,cat_cols = cat_cols,minmax_scale = T)

# Train the algorithm #
train_model <- train(data = converted_vars,training_epochs = 40,layer_structure = c(512,512,512),
                     input_drop = 0.95,seed = 42,learn_rate = 0.000001,
                     vae_layer = T)


# generate 50 different imputations per NA value #
# if there aren't category variables than cat_coalesce = F #
m <- 50
imputaions_sims <- complete(train_model,m = m,cat_coalesce = F,unscale = T)

# average all imputaions across m repeats #
sum.mat <- NULL
for (matrix in 1:length(imputaions_sims)) {
  if (is.null(sum.mat)) {
    sum.mat <- imputaions_sims[[matrix]][,1:(ncol(merged_data) - 9)]
  }else{
    sum.mat <- sum.mat + imputaions_sims[[matrix]][,1:(ncol(merged_data) - 9)]
  }
}

imputated_umi_replication <- sum.mat / m

merged_data <- merge(iso_data,clinical_replication,by = 'ID', all = F)
rownames(merged_data) <- merged_data[,1]  ; merged_data <- merged_data[,-1]
merged_data <- merged_data[,-match(table = colnames(merged_data),x = c('PID','CollNum_p','SampleID_p','Os3_Bulbar_dm',
                                                                       'Os3_Limb_dm','Os3_Other_dm','deltaFRS','Endpt_reason'))]
rownames(imputated_umi_replication) <- rownames(merged_data)


batch_replication <- sapply(rownames(imputated_umi_replication), function(x)(strsplit(x,'_')[[1]][2])) ; names(batch_replication) <- NULL
# check for confounders in sex or batch #



##########################################
#    Batch correct all isoMIRs data      #
##########################################

# ----------------------------------------------- #
#               For ALS patients data             #
# ----------------------------------------------- #

# comBat needs normalized counts so I use cpm #

cpmdata <- cpm(t(imputated_umi_replication),log=TRUE, normalized.lib.sizes=TRUE)

comdata <- ComBat(dat = cpmdata,batch = batch_replication)

# I reversed the log transformation by cpm in order to correct by DESeq correction #

revertdata = t(((2^t(comdata))*colSums(t(imputated_umi_replication))/1000000))

#########################################################
#     correct imputed UMI data by geometrical mean      #
#########################################################

geom_mean <- apply(revertdata, 1, geometric.mean)
DESeq_norm <- revertdata / geom_mean
correct_factor <- apply(DESeq_norm, 2, median)
norm_counts_all_isoMIRs <- t(t(revertdata) / correct_factor)


clinical_replication$Endpt_LV[!is.na(clinical_replication$Endpt_reason)] <- 1

norm_counts_all_isoMIRs[is.na( filtered_umi_mat_replication[,match(table = colnames(filtered_umi_mat_replication),x = colnames(norm_counts_all_isoMIRs))])] <- 0

corrected_umi_replication <- cbind.data.frame(t(norm_counts_all_isoMIRs),'outcome' = clinical_replication$Endpt_LV[match(colnames(revertdata),clinical_replication$ID)],
                                              'Survival.from.onset' = clinical_replication$SurvOs_mo[match(colnames(revertdata),clinical_replication$ID)],
                                              'Survival.from.enrolment' = clinical_replication$SurvColl_mo[match(colnames(revertdata),clinical_replication$ID)])
