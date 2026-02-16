##### clean workspace #####
rm(list=ls())
main_dir = "/home/aschmidt/R-Abbildungen/VR-Valentin/"
setwd(main_dir)
getwd()

# define some nice colors
color_background = '#ebebeb'
color_violin = '#d4d4d4'

# clear the console
cat("\014") 

# load some required packages
library(tcltk)
library(ggplot2)
library(dplyr)
library(ggpubr)
library(reshape2)
library(geepack)
library(emmeans)
library(cowplot)
library(data.table)

# Directory containing the subject folders
root_dir <- "/BIOMAG_DATA/Projects/GR_Valentin/data/processed_final/Round2"

# Loop over all subject folders in the main directory
subject_folders <- list.dirs(root_dir, full.names = TRUE, recursive = FALSE)

# Empty result data frame
data <- data.frame()

#test = data[data$ID == '20',]

# Iterate over all subject folders
for (folder in subject_folders) {
  # List all files in the folder
  srt_files <- list.files(folder, full.names = TRUE)
  
  #folder = subject_folders[15]
    
  # Check if any file contains 'MRT_prae.txt' in its name
  srt_file_exists <- any(grepl("_unvollstaendig.csv", srt_files))
  srt_files = srt_files[which(grepl("_unvollstaendig.csv", srt_files))]
  
  print(srt_files)
  
  # Check if the folder contains 'MST'
  if (srt_file_exists) {
    for(srt_file in srt_files){
      dummy = data.frame(read.csv(file=file.path(srt_file),sep = ';',dec = ','))
      
      # Nummer nach GR_
      # Split the path into parts
      path_parts <- strsplit(srt_file, "/")[[1]]
      
      # Get the ID — the folder just before the file name
      # That would be the second-to-last element
      ID <- as.numeric(path_parts[length(path_parts) - 1])
      
      # Nummer nach FRA_
      day <- sub(".*FRA_(\\d+).*", "\\1", srt_file)
      
      # extract wanted data and combine with meta informations
      dummy = dummy %>% mutate(ID,day,,file=srt_file) %>%
        dplyr::select(ID,Block=BlockNumber,Event=EventNumber,TSBs=Time.Since.Block.start,isHit,SeqKind=sequence,day)
      
      data = rbind(data,dummy)
    }
  }
}

# convert some variables into factors
data$ID  = as.factor(data$ID)
data$SeqKind  = as.factor(data$SeqKind)
data$day  = as.factor(data$day)
data = data %>% droplevels()

# just for check up reason
head(data)
tail(data)

# Create a master output directory
results_dir <- file.path(main_dir, "SRT-Results")
if (!dir.exists(results_dir)) {
  dir.create(results_dir, recursive = TRUE)
}
setwd(results_dir)

##### load meta data #####
# read table
df <- read.csv("/BIOMAG_DATA/Projects/GR_Valentin/data/processed_final/Round2/Datensatz_Round_2.csv",row.names=NULL,stringsAsFactors = TRUE,na.strings=c("","N.A.","null"),sep=';',dec = c(','))
df <- df %>%
  rename_with(~ sub("^Biomag_", "", .x), starts_with("Biomag_"))

df = df %>% dplyr::select(ID = PID,Group,Age=Untersuchung,Gender = gender,AES = AES_sum,EQ5D = EQ5D_health_status,FuglMayr = fuglmayrshort_sum,GDS = GDS_sum,MoCA = MoCa_sum,MORE = MORE_sum, TSS, NIHSS,
                               weight,height,glasses=Wears.glasses..finding.,
                               smoker,alc_behav = Finding.relating.to.alcohol.drinking.behavior..finding.,alc_num = number.of.alcohol.units.consumed.on.typical.drinking.day..observable.entity.,
                               drugs=Misuses.drugs..finding.,gambling = Gambling..finding.,HistDep = History.of.depression,HistPsych=History.of.psychiatric.disorder, alone = Lives.alone..finding.,MartStat = Maritual.status,Bildungsgrad)
df = df %>% mutate(BMI = weight / (height/100)^2)
df$BMI

# Umbennen der ID spalte
df$ID = as.factor(df$ID)
# Create a factor variable with labels
df$Group <- factor(df$Group, levels = c('A','B'), labels = c("A", "B"))
df$Gender <- factor(df$Gender, levels = c('Female', 'Male'), labels = c("f", "m"))
df$Bildungsgrad <- factor(df$Bildungsgrad, levels = c("8-10. Klasse", "Abitur", "Ausbildung", "Studium"), labels = c("8.-10. Klasse", "Abitur", "Ausbildung", "Studium"))

# MoCA von ID 35 ersetzen
df$MoCA[df$ID == '35'] <- mean(df$MoCA, na.rm = TRUE)
#df$MoCA[df$ID == '35'] <- 28

# kritische Grenzwerte definieren
kritisch <- df %>%
  mutate(
    AES_crit = AES > 20,
    GDS_crit = GDS >= 6,
    MoCA_crit = MoCA < 24,
    NIHSS_crit = NIHSS >= 5
  )

kritische_faelle <- df %>%
  mutate(
    AES_crit = AES > 20,
    GDS_crit = GDS >= 6,
    MoCA_crit = MoCA < 24,
    NIHSS_crit = NIHSS >= 5,
    # Summenspalte der kritischen Variablen
    crit_sum = AES_crit + GDS_crit + MoCA_crit + NIHSS_crit
  ) %>%
  # nur Fälle mit mindestens 2 kritischen Werten
  filter(crit_sum >= 2) %>%
  dplyr::select(ID, AES, GDS, MoCA, NIHSS,
                AES_crit, GDS_crit, MoCA_crit, NIHSS_crit,
                crit_sum)
# show the critical files
kritische_faelle


###### calculate the main target value in the SRT reward task ######
DataBlockMax = data %>% group_by(ID,Block,day,SeqKind) %>% summarise(maxTSBs = max(TSBs)*1000,
                                                                          minTSBs= min(TSBs)*1000,
                                                                          durTSBs = (max(TSBs) - min(TSBs))*1000,
                                                                          Hitratio=sum(isHit)/8,
                                                                          n = n()) %>% ungroup()
# make some basic concept checks
unique(DataBlockMax$ID[DataBlockMax$Hitratio > 1])

unique(as.data.frame(DataBlockMax[DataBlockMax$n > 8,])) # ID 20

as.data.frame(unique(DataBlockMax[DataBlockMax$Block > 120,])) # ID 9 
#DataBlockMax = DataBlockMax %>% filter(Block <= 160) # remove the overhead blocks
DataBlockMax = DataBlockMax %>% filter(Block <= 120) # cut all down to 120 Blocks

# get the number of removed datapoints
nrow_bf = nrow(DataBlockMax)
# filter out bad hitratio values
nrow_af_0er = nrow(DataBlockMax %>% dplyr::filter(Hitratio >= 1))
nrow_af_1er = nrow(DataBlockMax %>% dplyr::filter(Hitratio >= 0.875))
nrow_af_2er = nrow(DataBlockMax %>% dplyr::filter(Hitratio >= 0.75))
print(paste0('Proz remaining Datapoints 0 error:',round(nrow_af_0er/nrow_bf*100,2)))
print(paste0('Proz remaining Datapoints 1 error:',round(nrow_af_1er/nrow_bf*100,2)))
print(paste0('Proz remaining Datapoints 2 errors:',round(nrow_af_2er/nrow_bf*100,2)))
# chose no error ==> use Hitratio in models as confounder
# DataBlockMax = DataBlockMax %>% dplyr::filter(Hitratio >= 1) %>% droplevels()
print(nrow(DataBlockMax))

#### getting cooks distance into account ####
# calc cooks distance for all files
cookd_dat = DataBlockMax %>% group_by(ID,SeqKind,day) %>% summarise(CooksDist = cooks.distance(lm(maxTSBs~Block+1)),Block=Block,n = n())

# include Cooks Distance for each oberservation
DataBlockMax = DataBlockMax %>% dplyr::select(-n) %>% dplyr::left_join(cookd_dat,by = c('ID','SeqKind','Block','day'))

# make  a short wished plot
levels(DataBlockMax$ID)
ID_choice = '3'
t1 = DataBlockMax
t1 = t1 %>% filter(ID == ID_choice)
p = ggplot() +
  geom_point(data = t1, aes(x = Block, y = maxTSBs),colour='red') +
  geom_smooth(data = t1, aes(x = Block, y = maxTSBs),method = 'lm',colour='red',fullrange=TRUE) +
  ylab('ReactionTime (ms)') +
  facet_grid(SeqKind~day)

# filter out oberservations which influneces the estimation the most
nBefore = nrow(DataBlockMax)
DataBlockMax = DataBlockMax %>% filter(CooksDist <= 4/n)
nAfter = nrow(DataBlockMax)
print('remaining Datapoints:')
print(nAfter/nBefore)

t2 = DataBlockMax
t2 = t2 %>% filter(ID == ID_choice)
p = p + geom_point(data = t2, aes(x = Block, y = maxTSBs),colour='blue') +
  geom_smooth(data = t2, aes(x = Block, y = maxTSBs),method = 'lm',colour='blue',fullrange=TRUE) +
  ylab('ReactionTime (ms)') +
  stat_regline_equation(data = t2, aes(x = Block, y = maxTSBs,label =  paste(..eq.label.., sep = "~~~~")),colour='blue') + theme(text=element_text(size=12),legend.position = 'bottom')
plot(p)
png(paste0("SRT-CooksDistanceExample-ID-selected.png"), width = 50, height = 30, units = "cm", res = 300)
plot(p)
dev.off()

##### combine raw data with the meta data #####
GData = DataBlockMax %>% left_join(df,by='ID') %>% droplevels() %>% dplyr::ungroup() %>% dplyr::select(-contains('Anzahl'),-contains('Path'),-contains('vollstaendig'))
str(GData)
# remove data with no meta data until now ==> no grouplabel
print(unique(GData$ID[which(is.na(GData$Group))])) # missing behav data ==> das hat so seine Richtigkeit
GData <- GData %>% filter(!is.na(Group))

# Save as semicolon-separated CSV with decimal points as "."
write.table(GData, file = "SRT-DataBlockMax.csv", sep = ";", dec = ".", row.names = FALSE)

##### filter out unwanted data #####
# ID 2/7/13 just have one incomplete day file
GData <- GData %>%
  filter(!(
    (ID == '7' & day == '1') | # shorter file duration day 1
    (ID == '20' & day == '1') | # more than one file from different dates for day 1
    (ID == '20' & day == '2') | # really positiv slope (extrem outliers)
    (ID == '27' & day == '1') | # shorter file duration day 1
    (ID == '38' & day == '1') | # shorter file duration day 1
    (ID == '13' & day == '1') | # shorter file duration day 1 and strange pattern
    (ID == '2' & day == '1') | # shorter file duration day 1
     ID == '32' | # both days shorter duration
     ID == '35' |
     ID == '5' | # extrem high AES value
     ID == '33' | # bad fugl meyer
     ID == '4' | # bad fugl meyer
     ID == '9' # bad fugl meyer
  ))

##### check linear block assumption for maxTSBs #####
summary(GData$maxTSBs)

# Get factor levels
group_levels <- levels(GData$Group)

# Loop through each combination
for (gr in group_levels) {
  
  # Filter data
  sub_dat = GData %>%
    filter(Group == gr) %>%
    droplevels()
  
  # Skip if no data
  if (nrow(sub_dat) == 0) next
  
  # Determine unique facet levels
  n_IDs <- length(unique(sub_dat$ID))
  n_days <- length(unique(sub_dat$day))
  
  # Set dimensions: adjust height by number of rows and columns in facets
  width_cm <- 10 * n_days      # 10 cm per day
  height_cm <- 5 * n_IDs       # 5 cm per ID
  
  # Create plot
  color_pal_vec = c('#1a5276','#0e6655','#d4ac0d')
  p = ggplot(sub_dat, aes(x = Block, y = maxTSBs, color = SeqKind)) +
    geom_point() +
    facet_grid(ID ~ day, scales = 'free') +
    geom_smooth(method = 'lm',fullrange = TRUE,formula = y ~ poly(x, 1)) +
    scale_x_continuous(breaks = seq(0, 120, 20)) +
    theme(legend.position = 'bottom') +
    scale_color_manual(values = color_pal_vec)
  
  # Save plot
  file_name <- paste0("SRT-ID-BlockSlopes-", gr, ".png")
  png(file_name, width = width_cm, height = height_cm, units = "cm", res = 300)
  print(p)
  dev.off()
  
  cat("Saved:", file_name, " (", width_cm, "x", height_cm, "cm)\n")
}

##### make a behavedata overview ######
library(arsenal)

# select sub-data
test = GData 

# make behave table comparison
temp = test %>%
  dplyr::select(-Block, -maxTSBs, -minTSBs, -durTSBs, -Hitratio, -SeqKind, 
         -day, -CooksDist, -n) %>%
  unique() %>% droplevels() %>% dplyr::select(-ID)
temp <- temp %>%
  # Entferne Faktorvariablen mit nur einem Level
  dplyr::select(where(~ !(is.factor(.) && nlevels(.) <= 1))) %>%
  # Entferne numerische Spalten mit 0 Varianz (konstant)
  dplyr::select(where(~ !(is.numeric(.) && (is.na(sd(., na.rm = TRUE)) || sd(., na.rm = TRUE) == 0))))
str(temp) # quick controll
# create table
table_one <- tableby(Group ~ ., data = temp, control = tableby.control(chisq.correct = TRUE))
SumTable <- summary(table_one, title = paste("SummaryTable -"), text = TRUE)

# Speichere die Zusammenfassung als .txt statt .csv (wegen Formatierung)
write.csv2(as.data.frame(SumTable), row.names = FALSE,
           paste0("SRT-VRGame-Behavioral-Table.csv"))

##### make a correlation plot #####
gcor = test %>%
  dplyr::select(-Block, -maxTSBs, -minTSBs, -durTSBs, -Hitratio, -SeqKind, 
         -day, -CooksDist, -n) %>% unique() %>% droplevels() %>% dplyr::select(-ID)
gcor <- gcor %>%
  # Entferne Faktorvariablen mit nur einem Level
  dplyr::select(where(~ !(is.factor(.) && nlevels(.) <= 1))) %>%
  # Entferne numerische Spalten mit 0 Varianz (konstant)
  dplyr::select(where(~ !(is.numeric(.) && (is.na(sd(., na.rm = TRUE)) || sd(., na.rm = TRUE) == 0))))

# make all numeric
gcor = as.data.frame(lapply(gcor, function(x) as.numeric(x)))

p_mat_cor = psych::corr.test(gcor, method = 'spearman', adjust = 'none', use = 'pairwise.complete.obs')
gcor_matrix = p_mat_cor$r

# Dynamische Größe: 1.5 cm pro Variable, mindestens 15 cm, maximal 40 cm
num_vars <- ncol(gcor_matrix)
plot_size_cm <- min(max(1.5 * num_vars, 15), 40)

png(paste0("SRT-SpearmanCorrelationPlot.png"), units = "cm", res = 300, 
    width = plot_size_cm, 
    height = plot_size_cm)
corrplot::corrplot(gcor_matrix, p.mat = p_mat_cor$p, tl.cex = 0.75, tl.srt = 45, number.cex = 1)
dev.off()

write.csv2(gcor, row.names = FALSE,paste0("SRT-CorrelationTable.csv"))

##### create mean Pat vales over the Blocks #####
df0 = GData %>% group_by(ID,day,SeqKind,Group) %>% summarise(MmaxTSBs = mean(maxTSBs,na.rm = TRUE),
                                                                            MminTSBs = mean(minTSBs,na.rm = TRUE),
                                                                            MdurTSBs = mean(durTSBs,na.rm = TRUE),
                                                                            BlockSlopeMax = -coef(lm(maxTSBs~Block+1))[2],
                                                                            SE_BlockSlopeMax = {
                                                                              m <- lm(maxTSBs ~ Block+1)
                                                                              sqrt(diag(vcov(m)))[2]},
                                                                            nMax=coef(lm(maxTSBs~Block+1))[1],
                                                                           SE_nMax = {
                                                                             m <- lm(maxTSBs ~ Block+1)
                                                                             sqrt(diag(vcov(m)))[1]},
                                                                            BlockSlopeDur = -coef(lm(durTSBs~Block+1))[2],
                                                                            nDur=coef(lm(durTSBs~Block+1))[1],
                                                                            MHitratio = mean(Hitratio,na.rm = TRUE),
                                                                            maxBlock = max(Block), 
                                                                            numObs = n()) %>% ungroup()
df0 = df0 %>% mutate(MaxB120 = 120*-BlockSlopeMax+nMax,
                     DurB120 = 120*-BlockSlopeDur+nDur)
str(df0)

# Save as semicolon-separated CSV with decimal points as "."
write.table(df0, file = "SRT-DataBlockMax-PatMeans.csv", sep = ";", dec = ".", row.names = FALSE)

##### make simple t-test boxplots for choosen target variables
# Vector of target variables
target_vec = c('MmaxTSBs','MminTSBs','MdurTSBs','nMax','nDur', 'BlockSlopeMax','BlockSlopeDur','MaxB120','DurB120')
    
# Iterate over the target variables
for (target_var in target_vec) {
  
  #target_var = target_vec[1]
  # Create a subfolder named after the target variable (if it doesn't exist)
  subfolder_path <- file.path(target_var)
  if (!dir.exists(subfolder_path)) {
    dir.create(subfolder_path)
  }
  
  # Set working directory to the subfolder
  setwd(subfolder_path)
  
  ##### make Boxplot - for group comparison #####
  # Calculate the number of unique levels for the facet variables (day and SeqKind)
  n_tage = length(unique(df0$day))
  
  # Get the unique levels of VR_Spiel in the current subset
  Group_levels <- as.character(unique(df0$Group))
  
  # Create all pairwise combinations of those levels
  my_comparisons <- combn(Group_levels, 2, simplify = FALSE)
  
  # Define the color palette for the plots
  color_pal_vec <- c("midnightblue","darkgreen",'darkgoldenrod')
  
  # Create the plot for each target variable
  p = ggplot(data = df0, mapping = aes_string(x = 'Group', y = target_var, color = 'SeqKind')) +
    geom_boxplot(outlier.shape = NA) +
    geom_jitter(width = 0.1, height = 0.1, alpha = 0.7) +
    geom_text(aes(label = ID), hjust = -0.1, size = 4, alpha = 0.7) +
    xlab('Group') + 
    ylab(paste0(target_var, ' (ms)')) +
    facet_grid(SeqKind  ~ day, scales = 'fixed') +
    theme(legend.title = element_blank(), legend.position = "bottom") +
    stat_compare_means(method = 't.test', comparisons = my_comparisons, label = "p.format", paired = FALSE) +
    scale_color_manual(values = color_pal_vec)
  
  # Define the file name dynamically based on the current values of 'Gruppe', 'Hand', and 'target_var'
  file_name = paste0("Boxplot-Group-UnPairedTTest-", target_var, ".png")
  
  # Save the plot as a PNG file
  ggsave(file_name, plot = p, width = n_tage * 10, height = 30, dpi = 300,units = "cm")
  
  # Optionally, print a message or the plot to confirm it's saved
  print(paste("Saved plot:", file_name))
  
  ##### make Boxplot - for day comparison #####
  # Calculate the number of unique levels for the facet variables (day and SeqKind)
  n_groups = length(unique(df0$Group))
  
  # Get the unique levels of VR_Spiel in the current subset
  day_levels <- as.character(unique(df0$day))
  
  # Create all pairwise combinations of those levels
  my_comparisons <- combn(day_levels, 2, simplify = FALSE)

  # remove missing values from the day data , for the paired t-test
  temp = table(df0$ID,df0$day) 
  rm_names = rownames(temp)[rowSums(temp != 3) >= 1]
  test = df0[!(df0$ID %in% rm_names),] %>% droplevels()
  table(test$ID,test$day)
  
  # Create the plot for each target variable
  p = ggplot(data = test, mapping = aes_string(x = 'day', y = target_var, color = 'SeqKind')) +
    geom_boxplot(outlier.shape = NA) +
    geom_line(aes(group = ID), color = 'black', alpha = 0.5) +
    geom_point(alpha = 0.7) +
    geom_text(aes(label = ID), hjust = -0.1, size = 4, alpha = 0.7) +
    xlab('day') + 
    ylab(paste0(target_var, ' (ms)')) +
    facet_grid(SeqKind ~ Group, scales = 'fixed') +
    theme(legend.title = element_blank(), legend.position = "bottom") +
    stat_compare_means(method = 't.test', comparisons = my_comparisons, label = "p.format", paired = TRUE) +
    scale_color_manual(values = color_pal_vec)
  
  # Define the file name dynamically based on the current values of 'Gruppe', 'Hand', and 'target_var'
  file_name = paste0("Boxplot-days-Paired-TTest-", target_var, ".png")
  
  # Save the plot as a PNG file
  ggsave(file_name, plot = p, width = n_groups * 10, height = 30, dpi = 300,units = "cm")
  
  # Optionally, print a message or the plot to confirm it's saved
  print(paste("Saved plot:", file_name))
  
  # jump a directory back
  setwd("..")
}

##### check different GEE Models #####
# get plotting function
source("/home/aschmidt/R-Abbildungen/stroke_reward_reaction/R-Functions/plot_EMM_Effects.R")

# Vector of target variables
#target_vec = c( 'BlockSlopeMax','BlockSlopeDur','MmaxTSBs','MminTSBs','MdurTSBs','nMax','nDur')
target_vec = c('nMax', 'BlockSlopeMax','MaxB120','nDur', 'BlockSlopeDur','DurB120')

# Ensure output folder exists
output_dir <- "GEE-Results"
if (!dir.exists(output_dir)) {
  dir.create(output_dir)
}

# Iterate over each target variable
for (target in target_vec) {
  
  #target = target_vec[1]
  cat("\n\n##### Running models for target:", target, "#####\n")
  
  # Create subdirectory for this target
  target_output_dir <- file.path(output_dir, target)
  if (!dir.exists(target_output_dir)) {
    dir.create(target_output_dir, recursive = TRUE)
  }
  
  # Dynamically build formulas for OH group
  formulas <- list(
    as.formula(paste(target, "~ Group + day + SeqKind  + Age")),
    as.formula(paste(target, "~ Group * day * SeqKind  + Age")),
    as.formula(paste(target, "~ Group * day * SeqKind  + Age + MHitratio")),
    as.formula(paste(target, "~ Group * day * SeqKind  + Age + MHitratio + MoCA")),
    as.formula(paste(target, "~ Group * day * SeqKind  + Age + AES" )),
    as.formula(paste(target, "~ Group * day * SeqKind  + Age + TSS")),
    as.formula(paste(target, "~ Group * day * SeqKind  + Age + NIHSS")),
    as.formula(paste(target, "~ Group * day * SeqKind  + Age + MORE")),
    as.formula(paste(target, "~ Group * day * SeqKind  + Age + EQ5D")),
    as.formula(paste(target, "~ Group * day * SeqKind  + Age + TSS")),
    as.formula(paste(target, "~ Group * day * SeqKind  + Age + NIHSS + TSS")),
    as.formula(paste(target, "~ Group * day * SeqKind  + Age"))
  )
  
  # Prepare Base Data
  baseData <- df0 %>%
    left_join(df, by = c("ID",'Group')) %>%
    dplyr::select(-contains("Anzahl"), -contains("Path"), -contains("vollstaendig")) %>%
    droplevels() %>%
    ungroup()
      
  # Step 1: Extract all variable names from the formulas
  vars_in_formulas <- unique(unlist(
    lapply(formulas, function(f) all.vars(f))
  ))
  
  # Step 2: Ensure 'ID' is included for GEE grouping
  vars_in_formulas <- unique(c(vars_in_formulas, "ID",'SE_nMax','SE_BlockSlopeMax'))
  
  # Step 3: Subset baseData to only those columns
  baseData_subset <- baseData %>%
    dplyr::select(all_of(vars_in_formulas)) %>%
    na.omit() %>% droplevels()
  
  # Skip if too few data points
  if (nrow(baseData_subset) < 10) {
    cat("Skipping due to insufficient data\n")
    next
  }
  
  # Fit GEE models
  models <- list()
  # Correlation Structures (assuming same for all models)
  cor_structs <- rep("exchangeable", length(formulas))  # 7 formulas per group
  if(target == 'nMax'){w = 1/(baseData_subset$SE_nMax^2)}
  else{w = 1/(baseData_subset$SE_BlockSlopeMax^2)}
  for (i in seq_along(formulas)) {
    models[[i]] <- geeglm(formulas[[i]], id = ID, family = gaussian, corstr = cor_structs[i], data = baseData_subset)
    #models[[i]] <- lmerTest::lmer(update(formulas[[i]], . ~ . + (1 | ID)), data = baseData_subset)
  }
  
  # Extract QICs
  qics <- lapply(models, geepack::QIC)
  qic_vals <- sapply(qics, function(x) x[1])   # QIC
  qicu_vals <- sapply(qics, function(x) x[2])  # QICu
  cic_vals <- sapply(qics, function(x) x[4])   # CIC
  qlikeli_vals <- sapply(qics, function(x) x[3])   # quasie likelihood
  n_params <- sapply(models, function(m) length(coef(m)))  # Number of parameters
  
  # Output model summary
  summary_file <- file.path(target_output_dir, paste0("GEE_Model_Summary_", target, ".txt"))
  sink(summary_file)
  print('Overview Error rates: (not group specific)')
  print(paste0('Proz remaining Datapoints 0 error:',round(nrow_af_0er/nrow_bf*100,2)))
  print(paste0('Proz remaining Datapoints 1 error:',round(nrow_af_1er/nrow_bf*100,2)))
  print(paste0('Proz remaining Datapoints 2 errors:',round(nrow_af_2er/nrow_bf*100,2)))
  print('remaining Datapoints after cooks-distance clearing: (not group specific)')
  print(nAfter/nBefore)
  
  print('used IDs: ')
  print(unique(levels(baseData_subset$ID)))
  
  print(texreg::screenreg(models, custom.gof.rows = list(
    num.Parameters = n_params,
    CorrelationStructur = cor_structs,
    CIC = cic_vals,
    QIC = qic_vals,
    QICu = qicu_vals,
    Qlikeli = qlikeli_vals
  )))
  # Select best model
  # Normalize QIC to range [0, 1]
  qic_scaled <- (qic_vals - min(qic_vals)) / (max(qic_vals) - min(qic_vals))
  
  # Normalize parameter count to same scale
  params_scaled <- (n_params - min(n_params)) / (max(n_params) - min(n_params))
  
  # Weighted sum (e.g., 70% QIC, 30% model complexity)
  penalized_score <- 0.7 * qic_scaled + 0.3 * params_scaled
  best_model_index <- which.max(qlikeli_vals)
  best_model_index <- 4 
  best_model <- models[[best_model_index]]
  print('')
  print(cat("Selected Model:", best_model_index, "with QIC =", qic_vals[best_model_index], "\n"))
  print('')
  print('Best Model Summary: ')
  print(summary(best_model))
  print('')
  print('Variance Inflation Factor: lm-workaround')
  lm_model <- lm(formulas[[best_model_index]], data = baseData_subset)
  print(car::vif(lm_model))
  sink()
  
  # print selected model
  cat("Selected Model:", best_model_index, "with QIC =", qic_vals[best_model_index], "\n")
  
  ### plot confounder influence ###
  source('/home/aschmidt/R-Abbildungen/stroke_reward_reaction/R-Functions/plot_model_effect.R')
  # Variables that are NOT confounders
  non_confounders <- c('Group' ,"day","SeqKind")  # always in positions 2 and 3
  
  # All predictors in the model excluding the response
  all_preds <- all.vars(formula(best_model))[-1]
  
  # Only keep predictors that exist in df and are NOT non_confounders
  confounders <- setdiff(intersect(all_preds, c(names(df),'MHitratio') ), non_confounders)
  
  # Subfolder for saving plots
  sub_dir <- file.path(target_output_dir, "lin. confounder Influence")
  if (!dir.exists(sub_dir)) dir.create(sub_dir, recursive = TRUE)
  
  # Iterate over numeric confounders
  for (confounder in confounders) {
    
    # target_vars: confounder first, then non_confounders
    target_vars <- c(confounder, non_confounders[1:min(2, length(non_confounders))])
    
    # Create the plot
    p <- plot_model_effect(baseData_subset, best_model = best_model, target_vars = target_vars)
    
    # Filename
    ttt <- target_vars[-1]  # non-confounders for filename
    file_name <- file.path(sub_dir, paste0("GEE_Group_Plot_", confounder, "_", paste(ttt, collapse = "-"), ".png"))
    
    # Save the plot
    ggsave(filename = file_name, plot = p, width = 25, height = 20, units = "cm", dpi = 300)
    
    message("Plot saved: ", file_name)
  }
  
  # Plotting (adjust as needed per target)
  factor_list <- list(c("Group",'day'))
  #cont_value_list = list(Alter=c(60,80))
  cont_value_list = list()
  factor_group_list <- list(c("day","SeqKind"))
  facet_list <- list(c("day","SeqKind"))
  color_vec <- c("SeqKind")
  color_pal_vec <- c("midnightblue","darkgreen",'darkgoldenrod',"#633838","#636658","#638AA8")
  source("/home/aschmidt/R-Abbildungen/stroke_reward_reaction/R-Functions/plot_EMM_Effects.R")
  p <- plot_EMM_Effects(baseData_subset, best_model,factor_list,factor_group_list,cont_value_list,facet_list, color_vec, color_pal_vec,TRUE,TRUE)
  
  file_name <- file.path(target_output_dir, paste0("GEE_Group_Plot_", target, ".png"))
  ggsave(file_name, plot = p$plots[[1]],  width = p$pic_sizes[[1]][1], height = p$pic_sizes[[1]][2], units = "cm", dpi = 300)
  file_name_csv <- sub("\\.png$", "_contrasts.csv", file_name)
  write.csv(p$contrasts[[1]], file = file_name_csv, row.names = FALSE)
  file_name_csv <- sub("\\.png$", "_estimates.csv", file_name)
  write.csv(p$estimates[[1]], file = file_name_csv, row.names = FALSE)
  cat("Saved plot to", file_name, "\n")
  
  # Plotting (adjust as needed per target)
  factor_list <- list(c("day",'Group'))
  #cont_value_list = list(Alter=c(60,80))
  cont_value_list = list()
  factor_group_list <- list(c("Group","SeqKind"))
  facet_list <- list(c("Group","SeqKind"))
  color_vec <- c("SeqKind")
  p <- plot_EMM_Effects(baseData_subset, best_model,factor_list,factor_group_list,cont_value_list,facet_list, color_vec, color_pal_vec,TRUE,TRUE)
  
  file_name <- file.path(target_output_dir, paste0("GEE_day_Plot_", target, ".png"))
  ggsave(file_name, plot = p$plots[[1]],  width = p$pic_sizes[[1]][1], height = p$pic_sizes[[1]][2], units = "cm", dpi = 300)
  file_name_csv <- sub("\\.png$", "_contrasts.csv", file_name)
  write.csv(p$contrasts[[1]], file = file_name_csv, row.names = FALSE)
  file_name_csv <- sub("\\.png$", "_estimates.csv", file_name)
  write.csv(p$estimates[[1]], file = file_name_csv, row.names = FALSE)
  cat("Saved plot to", file_name, "\n")
  
  # Plotting (adjust as needed per target)
  factor_list <- list(c("SeqKind",'Group',"day"))
  #cont_value_list = list(Alter=c(60,80))
  cont_value_list = list()
  factor_group_list <- list(c("Group","day"))
  facet_list <- list(c("Group","day"))
  color_vec <- c("Group")
  color_pal_vec = c("#4B2E83", "#7B3F00", "#283663","#2E834B", "#00437B","#832838","#63D88F","#8F3663","#383663","#583663","#A83663","#283883","#D83663", "#663828","#638F36","#633838","#636658","#638AA8")
  p <- plot_EMM_Effects(baseData_subset, best_model,factor_list,factor_group_list,cont_value_list,facet_list, color_vec, color_pal_vec,TRUE,TRUE)
  
  file_name <- file.path(target_output_dir, paste0("GEE_SeqKind_Plot_", target, ".png"))
  ggsave(file_name, plot = p$plots[[1]],  width = p$pic_sizes[[1]][1], height = p$pic_sizes[[1]][2], units = "cm", dpi = 300)
  file_name_csv <- sub("\\.png$", "_contrasts.csv", file_name)
  write.csv(p$contrasts[[1]], file = file_name_csv, row.names = FALSE)
  file_name_csv <- sub("\\.png$", "_estimates.csv", file_name)
  write.csv(p$estimates[[1]], file = file_name_csv, row.names = FALSE)
  cat("Saved plot to", file_name, "\n")
}


##### get offline learning performance values #####
# calc day 1 B120 difference to D2 nMax
temp = df0 %>% group_by(ID,SeqKind,Group) %>% summarize(Diff_D1B120_D2nMax = MaxB120[1]-nMax[2],
                                                        Diff_D1B120_D2nDur = DurB120[1]-nDur[2])
# Ensure output folder exists
output_dir <- "GEE-Results"
if (!dir.exists(output_dir)) {
  dir.create(output_dir)
}

target = 'Diff_D1B120_D2nMax'
cat("\n\n##### Running models for target:", target, "#####\n")

# Create subdirectory for this target
target_output_dir <- file.path(output_dir, target)
if (!dir.exists(target_output_dir)) {
  dir.create(target_output_dir, recursive = TRUE)
}

# Dynamically build formulas for OH group
formulas <- list(
  as.formula(paste(target, "~ Group + SeqKind + Age")),
  as.formula(paste(target, "~ Group * SeqKind + Age")),
  as.formula(paste(target, "~ Group * SeqKind + Age + MoCA")),
  as.formula(paste(target, "~ Group * SeqKind + Age + AES" )),
  as.formula(paste(target, "~ Group * SeqKind + Age + TSS")),
  as.formula(paste(target, "~ Group * SeqKind + Age + NIHSS")),
  as.formula(paste(target, "~ Group * SeqKind + Age + MORE")),
  as.formula(paste(target, "~ Group * SeqKind + Age + EQ5D")),
  as.formula(paste(target, "~ Group * SeqKind + Age + NIHSS * TSS + MORE")),
  as.formula(paste(target, "~ Group * SeqKind + Age + MoCA + NIHSS + TSS"))
)

# Prepare Base Data
baseData <- temp %>%
  left_join(df, by = c("ID",'Group')) %>%
  dplyr::select(-contains("Anzahl"), -contains("Path"), -contains("vollstaendig")) %>%
  droplevels() %>%
  ungroup()

# Step 1: Extract all variable names from the formulas
vars_in_formulas <- unique(unlist(
  lapply(formulas, function(f) all.vars(f))
))

# Step 2: Ensure 'ID' is included for GEE grouping
vars_in_formulas <- unique(c(vars_in_formulas, "ID"))

# Step 3: Subset baseData to only those columns
baseData_subset <- baseData %>%
  dplyr::select(all_of(vars_in_formulas)) %>%
  na.omit() %>% droplevels()

# Skip if too few data points
if (nrow(baseData_subset) < 10) {
  cat("Skipping due to insufficient data\n")
  next
}

# get a short overview of files in each group
df_ID_summary <- baseData_subset %>%
  group_by(Group) %>%
  summarise(
    n_files = n_distinct(ID),           # Anzahl eindeutiger IDs pro Gruppe
    IDs = paste(unique(ID), collapse=", ")  # IDs als kommagetrennte Liste
  )


# Fit GEE models
models <- list()
# Correlation Structures (assuming same for all models)
cor_structs <- rep("independence", length(formulas))  # 7 formulas per group
for (i in seq_along(formulas)) {
  models[[i]] <- geeglm(formulas[[i]], id = ID, family = gaussian, corstr = cor_structs[i], data = baseData_subset)
  #models[[i]] <- lmerTest::lmer(update(formulas[[i]], . ~ . + (1 | ID) + (1 | SeqKind)), data = baseData_subset)
}
texreg::screenreg(models, include.gof = FALSE)
print(texreg::screenreg(models))


# Extract QICs
qics <- lapply(models, geepack::QIC)
qic_vals <- sapply(qics, function(x) x[1])   # QIC
qicu_vals <- sapply(qics, function(x) x[2])  # QICu
cic_vals <- sapply(qics, function(x) x[4])   # CIC
qlikeli_vals <- sapply(qics, function(x) x[3])   # quasie likelihood
n_params <- sapply(models, function(m) length(coef(m)))  # Number of parameters

# Output model summary
summary_file <- file.path(target_output_dir, paste0("GEE_Model_Summary_", target, ".txt"))
sink(summary_file)
print('Overview Error rates: (not group specific)')
print(paste0('Proz remaining Datapoints 0 error:',round(nrow_af_0er/nrow_bf*100,2)))
print(paste0('Proz remaining Datapoints 1 error:',round(nrow_af_1er/nrow_bf*100,2)))
print(paste0('Proz remaining Datapoints 2 errors:',round(nrow_af_2er/nrow_bf*100,2)))
print('remaining Datapoints after cooks-distance clearing: (not group specific)')
print(nAfter/nBefore)

print('used IDs: ')
print(df_ID_summary)

print(texreg::screenreg(models, custom.gof.rows = list(
  num.Parameters = n_params,
  CorrelationStructur = cor_structs,
  CIC = cic_vals,
  QIC = qic_vals,
  QICu = qicu_vals,
  Qlikeli = qlikeli_vals
)))
# Select best model
# Normalize QIC to range [0, 1]
qic_scaled <- (qic_vals - min(qic_vals)) / (max(qic_vals) - min(qic_vals))

# Normalize parameter count to same scale
params_scaled <- (n_params - min(n_params)) / (max(n_params) - min(n_params))

# Weighted sum (e.g., 70% QIC, 30% model complexity)
penalized_score <- 0.7 * qic_scaled + 0.3 * params_scaled
best_model_index <- which.max(qlikeli_vals)
# best_model_index <- 9 
best_model <- models[[best_model_index]]
print('')
print(cat("Selected Model:", best_model_index, "with QIC =", qic_vals[best_model_index], "\n"))
print('')
print('Best Model Summary: ')
print(summary(best_model))
print('')
print('Variance Inflation Factor: lm-workaround')
lm_model <- lm(formulas[[best_model_index]], data = baseData_subset)
print(car::vif(lm_model))
sink()

# print selected model
cat("Selected Model:", best_model_index, "with QIC =", qic_vals[best_model_index], "\n")

### plot confounder influence ###
source('/home/aschmidt/R-Abbildungen/stroke_reward_reaction/R-Functions/plot_model_effect.R')
# Variables that are NOT confounders
non_confounders <- c('Group' ,"SeqKind")  # always in positions 2 and 3

# All predictors in the model excluding the response
all_preds <- all.vars(formula(best_model))[-1]

# Only keep predictors that exist in df and are NOT non_confounders
confounders <- setdiff(intersect(all_preds, c(names(df),'MHitratio') ), non_confounders)

# Subfolder for saving plots
sub_dir <- file.path(target_output_dir, "lin. confounder Influence")
if (!dir.exists(sub_dir)) dir.create(sub_dir, recursive = TRUE)

# Iterate over numeric confounders
for (confounder in confounders) {
  
  # target_vars: confounder first, then non_confounders
  target_vars <- c(confounder, non_confounders[1:min(2, length(non_confounders))])
  
  # Create the plot
  p <- plot_model_effect(baseData_subset, best_model = best_model, target_vars = target_vars)
  
  # Filename
  ttt <- target_vars[-1]  # non-confounders for filename
  file_name <- file.path(sub_dir, paste0("GEE_Group_Plot_", confounder, "_", paste(ttt, collapse = "-"), ".png"))
  
  # Save the plot
  ggsave(filename = file_name, plot = p, width = 25, height = 20, units = "cm", dpi = 300)
  
  message("Plot saved: ", file_name)
}

# Plotting (adjust as needed per target)
factor_list <- list(c("Group"))
cont_value_list = list()
factor_group_list <- list(c("SeqKind"))
facet_list <- list(c("SeqKind"))
color_vec <- c("SeqKind")
color_pal_vec <- c("midnightblue","darkgreen",'darkgoldenrod',"#633838","#636658","#638AA8")
source("/home/aschmidt/R-Abbildungen/stroke_reward_reaction/R-Functions/plot_EMM_Effects.R")
p <- plot_EMM_Effects(baseData_subset, best_model,factor_list,factor_group_list,cont_value_list,facet_list, color_vec, color_pal_vec,TRUE,TRUE)

file_name <- file.path(target_output_dir, paste0("GEE_Group_Plot_", target, ".png"))
ggsave(file_name, plot = p$plots[[1]],  width = p$pic_sizes[[1]][1], height = p$pic_sizes[[1]][2], units = "cm", dpi = 300)
file_name_csv <- sub("\\.png$", "_contrasts.csv", file_name)
write.csv(p$contrasts[[1]], file = file_name_csv, row.names = FALSE)
file_name_csv <- sub("\\.png$", "_estimates.csv", file_name)
write.csv(p$estimates[[1]], file = file_name_csv, row.names = FALSE)
cat("Saved plot to", file_name, "\n")

# Plotting (adjust as needed per target)
factor_list <- list(c("SeqKind",'Group'))
cont_value_list = list()
factor_group_list <- list(c("Group"))
facet_list <- list(c("Group"))
color_vec <- c("Group")
color_pal_vec = c("#4B2E83", "#7B3F00", "#283663","#2E834B", "#00437B","#832838","#63D88F","#8F3663","#383663","#583663","#A83663","#283883","#D83663", "#663828","#638F36","#633838","#636658","#638AA8")
p <- plot_EMM_Effects(baseData_subset, best_model,factor_list,factor_group_list,cont_value_list,facet_list, color_vec, color_pal_vec,TRUE,TRUE)

file_name <- file.path(target_output_dir, paste0("GEE_SeqKind_Plot_", target, ".png"))
ggsave(file_name, plot = p$plots[[1]],  width = p$pic_sizes[[1]][1], height = p$pic_sizes[[1]][2], units = "cm", dpi = 300)
file_name_csv <- sub("\\.png$", "_contrasts.csv", file_name)
write.csv(p$contrasts[[1]], file = file_name_csv, row.names = FALSE)
file_name_csv <- sub("\\.png$", "_estimates.csv", file_name)
write.csv(p$estimates[[1]], file = file_name_csv, row.names = FALSE)
cat("Saved plot to", file_name, "\n")