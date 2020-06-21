## LOADING LIBRARIES
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(caTools)) install.packages("caTools", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(class)) install.packages("class", repos = "http://cran.us.r-project.org")
if(!require(mlbench)) install.packages("mlbench", repos = "http://cran.us.r-project.org")
if(!require(glmnet)) install.packages("glmnet", repos = "http://cran.us.r-project.org")
if(!require(nnet)) install.packages("nnet", repos = "http://cran.us.r-project.org")
if(!require(DescTools)) install.packages("DescTools", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")


## DATA WRANGLING
#Read in the file, which is tab-delimited
data.full <-
  read.delim("songDb.tsv", header = TRUE, sep = "\t")

# Remove columns that don't serve as features (the Spotify URI, 
# The track reference, the full URL, etc.)
datax <-
  subset(data.full,
         select = -c(Uri, Ref_Track, URL_features, Type, ID, Name))

# Identify each song by its name (by changing the row names to song names)
rownames(datax) <- make.names(data.full$Name, unique = TRUE)
# Ensure the time signature is numeric, rather than a factor
datax$time_signature <- as.numeric(datax$time_signature)

# Tempo should also be numeric
datax$Tempo <- as.numeric(datax$Tempo)

as_tibble(datax)
summary(datax)

# FUNCTIONS and METHODS
# Test and Train Sets
# Let’s also define a function that makes it easy to create our train and test sets. 
# (Storing both, the full train and test sets as well as the separate X and Y data frames for each might seem redundant but it will make things just a bit easier down the road)

get_train_test <- function(split_ratio, data) {
  results <- list()
  
  split.index <- sample.split(seq_len(nrow(data)), split_ratio)
  
  results$data.train <- data[split.index, ]
  results$data.test <- data[!split.index, ]
  
  results$X.train <-
    results$data.train %>% select(-Genre) %>% as.matrix()
  results$Y.train <- results$data.train$Genre
  
  results$X.test <-
    results$data.test %>% select(-Genre) %>% as.matrix()
  results$Y.test <- results$data.test$Genre
  return(results)
}

knn_function <- function(data, genres) {
  data.sub <- data[data$Genre %in% genres, ]
  data.sub$Genre <- droplevels(data.sub$Genre)
  
  set.seed(101)
  # Create an empty data frame to store the predictions and the actual labels
  classifications <- data.frame(pred = factor(), actual = factor())
  # Use K-fold cross validation
  K = 5
  for (k in 1:K) {
    # shuffle the data
    res <- get_train_test(0.8, data.sub)
    fit.knn <-
      knn(
        train = res$X.train,
        test = res$X.test,
        cl = res$Y.train
      )
    classifications <-
      rbind(classifications,
            data.frame(pred = fit.knn, actual = res$Y.test))
  }
  confusionMatrix(classifications$pred, classifications$actual)
}

dtree_function <- function(data, genres) {
  data.sub <- data[data$Genre %in% genres,]
  data.sub$Genre <- droplevels(data.sub$Genre)
  res <- get_train_test(0.8, data.sub)
  
  # Decision Tree
  set.seed(103)
  fit.dtree <-
    train(
      Genre ~ .,
      data = res$data.train,
      method = "rpart",
      parms = list(split = "information")
    )
  
  Y.pred.dtree <-
    predict(fit.dtree, newdata = data.frame(res$X.test), type = "raw")
  confusionMatrix(Y.pred.dtree, res$Y.test)
}

# PILOT RESULTS
#Let's sample 10 genres at random from the dataset and see what our accuracy is

set.seed(3)
genres <- sample(levels(datax$Genre), 10)
knn_function(datax, genres)$overall["Accuracy"]
dtree_function(datax, genres)$overall["Accuracy"]

#Here's the histogram of the frequency distribution of the accuracy
set.seed(1)
distributionkNN <- replicate(100,{
  genres <- sample(levels(datax$Genre), 10)
  knn_function(datax, genres)$overall["Accuracy"]
})

hist(distributionkNN)

#Let's select 10 genres that we know to be sufficiently different from each other in terms of their genome composition and see if that increases the accuracy much.
genres <-
  list(
    "canadianpop",
    "electronica",
    "rock",
    "modernblues",
    "r&b",
    "polishblackmetal",
    "videogamemusic",
    "irishfolk",
    "koreanpop",
    "hiphop"
  )
knn_function(datax, genres)$overall["Accuracy"]
dtree_function(datax, genres)$overall["Accuracy"]

#kNN Model
genres <-
  list(
    "canadianpop",
    "electronica",
    "rock",
    "modernblues",
    "r&b",
    "polishblackmetal",
    "videogamemusic",
    "irishfolk",
    "koreanpop",
    "hiphop"
  )
knn_function(datax, genres)


#the Decision Tree model
genres <-
  list(
    "canadianpop",
    "electronica",
    "rock",
    "modernblues",
    "r&b",
    "polishblackmetal",
    "videogamemusic",
    "irishfolk",
    "koreanpop",
    "hiphop"
  )
dtree_function(datax, genres)

# TRYING TO IMPROVE ACCURACY

# most popular genres
descending <- datax %>% group_by(Genre) %>% summarise(n = n()) %>% arrange(desc(n))
descending

#Top 30 most popular genres
genre_top30 <- descending$Genre[1:30]
new_data <- filter(datax, Genre %in% genre_top30)

knn_function(new_data, genre_top30)$overall["Accuracy"]


# trend in this accuracy as we pick the top 50 to the top 10 most common genres.
seqx <- 50:10
top_trend <- function(x){
  genre_topx <- descending$Genre[1:x]
  new_datax <- filter(datax, Genre %in% genre_topx)
  knn_function(new_datax, genre_topx)$overall["Accuracy"]
}

accuracy_trend <- sapply(seqx, top_trend)
ggplot(data.frame(accuracy_trend), aes(x = seqx, y = accuracy_trend)) + geom_point() + geom_line()


#overlapping genres:
genre_grouped <- datax %>% group_by(Genre) %>%
  summarise(dance = mean(Danceability),
            energy = mean(Energy),
            key = mean(Key),
            loudness = mean(Loudness),
            mode = mean(Mode),
            speechness = mean(Speechness),
            acousticness = mean(Acousticness),
            intstrumentalness = mean(Instrumentalness),
            liveness = mean(Liveness),
            valence = mean(Valence),
            tempo = mean(Tempo),
            duration = mean(Duration_ms),
            Time = mean(time_signature)) %>%
  arrange(Genre)

genre_grouped

print(genre_grouped[c(which(genre_grouped$Genre == "swedishdeathmetal"), 
                      which(genre_grouped$Genre == "polishblackmetal")), ], width = Inf)
print(genre_grouped[c(which(genre_grouped$Genre == "indonesianpoppunk"), 
                      which(genre_grouped$Genre == "popemo")), ], width = Inf)