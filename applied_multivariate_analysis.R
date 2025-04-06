library(factoextra)
library(dplyr)
library(ggplot2)
library(factoextra)
library(FactoMineR)
wildcard = read.csv("batting_seasons_integration.csv") # offensive statistics 
anyNA(wildcard)
# cleaning data
dim(wildcard)
wildcard = wildcard[-895,]
anyNA(wildcard)

wildcard = wildcard %>% 
  filter(Season != 2020) # removing observations from 2020 

data_subset = wildcard[c(5:35)] # subset of numerical variables

data_final = data_subset[,-1] # removing 'GP' variable
data_final[] = lapply(data_final, as.numeric) # changing data to numeric

str(data_final)
dim(data_final) # n = 864, p = 30
reference = wildcard[,c(2:3)] # reference data frame of teams and years
set.seed(2002)
data_scaled = scale(data_final) # scaling data for k-means
wss = (nrow(data_scaled)-1)*sum(apply(data_scaled,2,var)) # calculating within-cluster variation
for (i in 2:10) wss[i] <- sum(kmeans(data_scaled, # trying 10 clusters
                                     centers=i)$withinss)
plot(1:10, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares", main = "Elbow Plot") # keeping 7 clusters
fit_initial = kmeans(data_final, 7, nstart=30) # k-means cluster initializing with 30
library(cluster)
set.seed(2002)
clusplot(data_scaled, fit_initial$cluster, color=TRUE, shade=TRUE,
         labels=2, lines=0, main = "PCA Plot of K-Means Cluster") # visualizing clusters
cluster1 = data_final[which(data_final$fit_initial.cluster=='1'),] # separating clusters into individual variables
cluster2 = data_final[which(data_final$fit_initial.cluster=='2'),]
cluster3 = data_final[which(data_final$fit_initial.cluster=='3'),]
cluster4 = data_final[which(data_final$fit_initial.cluster=='4'),]
cluster5 = data_final[which(data_final$fit_initial.cluster=='5'),]
cluster6 = data_final[which(data_final$fit_initial.cluster=='6'),]
cluster7 = data_final[which(data_final$fit_initial.cluster=='7'),]
reference = data.frame(reference, fit_initial$cluster)  # adding column of cluster assignments to reference data frame
teams1 = reference[which(reference$fit_initial.cluster=='1'),] # creating individual variables for each clusterings 
teams2 = reference[which(reference$fit_initial.cluster=='2'),]
teams3 = reference[which(reference$fit_initial.cluster=='3'),]
teams4 = reference[which(reference$fit_initial.cluster=='4'),]
teams5 = reference[which(reference$fit_initial.cluster=='5'),]
teams6 = reference[which(reference$fit_initial.cluster=='6'),]
teams7 = reference[which(reference$fit_initial.cluster=='7'),]
clusters = rbind(teams1, teams2, teams3, teams4, teams5, teams6, teams7) # reference data frame for all clusters
summary(teams1['Season'])

min(fit_initial$withinss)

print(fit_initial)
set.seed(2002)
data_hclust = data_scaled
dm = dist(data_hclust,method="euclidean")
hclust_data <- hclust(dm, method="complete")
plot(hclust_data)
par(mfrow = c(2,1))
plot(cut(as.dendrogram(hclust_data), h=6)$lower[[2]])
plot(cut(as.dendrogram(hclust_data), h=9)$lower[[2]])

# cut at h=6
reference[c(577, 667, 744, 637, 757, 507, 581, 628, 508, 659), ]


# cut at h=9
reference[c(447, 790, 815, 430, 417, 477, 779, 792, 540, 642, 820), ]
pca <- prcomp(data_final, scale = TRUE)
summary(pca)
fviz_screeplot(pca, ncp=10)
```
```{r}
pca_var <- get_pca_var(pca)
pca_var
pca_var$contrib
pca_var$coord
par(mfrow = c(2,1))
fviz_contrib(pca, choice = "var", axes = 1)
fviz_contrib(pca, choice = "var", axes = 2)

fviz_pca_var(pca, col.var="contrib")
for (col_name in names(data_final)) {
  # Check if the column is numeric
  if (is.numeric(data_final[[col_name]])) {
    # Generate the Q-Q plot
    qqplot <- ggplot(data_final, aes(sample = .data[[col_name]])) +
      stat_qq() +
      stat_qq_line(color = "blue") +
      ggtitle(paste("Q-Q Plot of", col_name)) +
      xlab("Theoretical Quantiles") +
      ylab("Sample Quantiles")
    
    # Display the Q-Q plot
    print(qqplot)
  }
}
pitching = read.csv("pitching_seasons_wildcard.csv")
anyNA(pitching)
# cleaning data
pitching = pitching[-895,]
anyNA(pitching)
pitching = pitching %>% 
  filter(Season != 2020) # removing observations from 2020 
pitching_final = pitching[c(6:33)] # subset of numerical variables
pitching_final[] = lapply(pitching_final, as.numeric) # changing data to numeric
pitching_reference = pitching[,c(2:3)]
master_df = cbind(data_final, pitching_final) # concatenating data frames
master_df = master_df[,-c(31:33,35)]

# renaming duplicative columns 
colnames(master_df)[c(36:37, 39:43)] <- c(
  "Hits allowed", 
  "Runs allowed", 
  "Home runs allowed", 
  "Walks allowed", 
  "Intentional walks allowed", 
  "Strikeouts Pitchers", 
  "HBP allowed"
)
df_scaled = scale(master_df) # scaling data for k-means
wss_2 = (nrow(df_scaled)-1)*sum(apply(df_scaled,2,var)) # calculating within-cluster variation
for (i in 2:10) wss_2[i] <- sum(kmeans(df_scaled, # trying 10 clusters
                                       centers=i)$withinss)
plot(1:10, wss_2, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares", main = "Elbow Plot for No. of Clusters") # keeping 6 clusters
fit_second = kmeans(master_df, 6, nstart=30) # k-means cluster
clusplot(df_scaled, fit_second$cluster, color=TRUE, shade=TRUE,
         labels=2, lines=0, main = "PCA Plot of K-Means Cluster") # visualizing clusters
cluster_df1 = master_df[which(master_df$fit_second.cluster=='1'),] # separating clusters into individual variables
cluster_df2 = master_df[which(master_df$fit_second.cluster=='2'),]
cluster_df3 = master_df[which(master_df$fit_second.cluster=='3'),]
cluster_df4 = master_df[which(master_df$fit_second.cluster=='4'),]
cluster_df5 = master_df[which(master_df$fit_second.cluster=='5'),]
cluster_df6 = master_df[which(master_df$fit_second.cluster=='6'),]
pitching_reference = data.frame(pitching_reference, fit_second$cluster)  # adding column of cluster assignments to reference data frame
dfclust1 = pitching_reference[which(pitching_reference$fit_second.cluster=='1'),] # creating individual variables for each clusterings 
dfclust2 = pitching_reference[which(pitching_reference$fit_second.cluster=='2'),]
dfclust3 = pitching_reference[which(pitching_reference$fit_second.cluster=='3'),]
dfclust4 = pitching_reference[which(pitching_reference$fit_second.cluster=='4'),]
dfclust5 = pitching_reference[which(pitching_reference$fit_second.cluster=='5'),]
dfclust6 = pitching_reference[which(pitching_reference$fit_second.cluster=='6'),]
dfclusters = rbind(dfclust1, dfclust2, dfclust3, dfclust4, dfclust5, dfclust6) # reference data frame for all clusters
df1 = read.csv("50_years_batting.csv")
df2 = read.csv("50_years_pitching.csv")

anyNA(df1)
anyNA(df2)

# removing observations from shortened seasons
df1 = df1 %>% 
  filter(!(Season %in% c(1981, 1994, 1995, 2020)))
df2 = df2 %>% 
  filter(!(Season %in% c(1981, 1994, 1995, 2020)))

# combining data sets
concat_df = cbind(df1, df2)

# creating subsets of the data
reference_df = concat_df[c(2:3)] # reference data frame with categorical data
concat_df = concat_df[,-c(1:5,8,36:43,45)] # removing unnecessary columns for analysis

# renaming duplicative column names
colnames(concat_df)[c(35:36, 38:42,46,53)] = c(
  "HA", 
  "RA", 
  "HRA", 
  "WA", 
  "IBA", 
  "SO P", 
  "HBPA",
  "ERA +",
  "SO/BB"
) # renaming duplicative column names

final_df = concat_df[,-c(1:2,17:21,29:30,46:53)] # keeping counting statistics only
final_df = sweep(final_df,1,unlist(df1[,5]),"/") # making features 'per game'
correlations = cor(final_df)
correlations = round(correlations, digits=2)
corrplot(correlations, method="shade", shade.col=NA, tl.cex=0.5, tl.col="black", title = "Correlation Matrix of Features")
for (col_name in names(final_df)) {
  # Check if the column is numeric
  if (is.numeric(final_df[[col_name]])) {
    # Generate the Q-Q plot
    qqplot <- ggplot(final_df, aes(sample = .data[[col_name]])) +
      stat_qq() +
      stat_qq_line(color = "blue") +
      ggtitle(paste("Q-Q Plot of", col_name)) +
      xlab("Theoretical Quantiles") +
      ylab("Sample Quantiles")
    
    # Display the Q-Q plot
    print(qqplot)
  }
  data_scaled = scale(final_df) # scaling data for k-means
  wss = (nrow(data_scaled)-1)*sum(apply(data_scaled,2,var)) # calculating within-cluster variation
  for (i in 2:10) wss[i] <- sum(kmeans(data_scaled, # trying 10 clusters
                                       centers=i)$withinss)
  plot(1:10, wss, type="b", xlab="Number of Clusters",
       ylab="Within groups sum of squares", main = "Elbow Plot") # keeping 5 clusters
  fit_initial = kmeans(final_df, 5, nstart=30) # k-means cluster initializing with 30
  clusplot(data_scaled, fit_initial$cluster, color=TRUE, shade=TRUE,
           labels=2, lines=0, main = "PCA Plot of K-Means Cluster") # visualizing clusters
  cluster1 = final_df[which(final_df$fit_initial.cluster=='1'),] 
  cluster2 = final_df[which(final_df$fit_initial.cluster=='2'),]
  cluster3 = final_df[which(final_df$fit_initial.cluster=='3'),]
  cluster4 = final_df[which(final_df$fit_initial.cluster=='4'),]
  cluster5 = final_df[which(final_df$fit_initial.cluster=='5'),]
  reference_df = data.frame(reference_df, fit_initial$cluster)  # adding column of cluster assignments to reference data frame
  teams1 = reference_df[which(reference_df$fit_initial.cluster=='1'),] # creating individual variables for each clustering
  teams2 = reference_df[which(reference_df$fit_initial.cluster=='2'),]
  teams3 = reference_df[which(reference_df$fit_initial.cluster=='3'),]
  teams4 = reference_df[which(reference_df$fit_initial.cluster=='4'),]
  teams5 = reference_df[which(reference_df$fit_initial.cluster=='5'),]
  clusters = rbind(teams1, teams2, teams3, teams4, teams5) # reference data frame for all clusters
  final_df = final_df[,-c(37)]
  pca = prcomp(final_df, scale = TRUE)
  fviz_screeplot(pca, ncp=10)
  summary(pca)
  pca_var = get_pca_var(pca)
  pca_var$contrib
  pca_var$coord
  par(mfrow = c(2,1))
  fviz_contrib(pca, choice = "var", axes = 1)
  fviz_contrib(pca, choice = "var", axes = 2)
  fviz_pca_var(pca, col.var="contrib")
  # Assuming final_df is your dataset with 36 numeric variables
  gmm_model <- Mclust(final_df)
  
  # View the summary of the model
  summary(gmm_model)
  # Cluster assignments
  clusters <- gmm_model$classification
  
  # Membership probabilities
  membership_probs <- gmm_model$z
  # Perform PCA for visualization
  pca_result <- prcomp(final_df, scale. = TRUE)
  
  # Create a data frame for plotting
  plot_data <- data.frame(PC1 = pca_result$x[,1], PC2 = pca_result$x[,2], Cluster = as.factor(clusters))
  
  # Plot using ggplot2
  library(ggplot2)
  ggplot(plot_data, aes(x = PC1, y = PC2, color = Cluster)) +
    geom_point() +
    theme_minimal() +
    labs(title = "GMM Clustering with PCA")