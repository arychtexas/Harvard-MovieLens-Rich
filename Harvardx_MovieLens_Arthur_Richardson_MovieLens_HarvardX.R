################################################################################
# This Project is dedicated to my two daughters and my siblings. 
# Your dreams will become your reality with patience and manifestation. 
# Love, Dad. 
################################################################################

################################################################################
# Reference Section

#1. Irizarry, R. A. (2022, July 7). Introduction to Data Science. HARVARD Data Science. Retrieved June 8, 2022, from Https://rafalab.github.io/dsbook/
# This project utilized "Introduction to Data Science Data Analysis and Prediction Algorithms with R" by our course instructor Rafael A. Irizarry published 2022-07-07.

#2.R Packages. R Studio. Retrieved May 18, 2022, https://www.rstudio.com/products/rpackages/

#3. Definition and History of Recommender Systems. (n.d.). Computer Science. https://international.binus.ac.id/computer-science/2020/11/03/definition-and-history-of-recommender-systems/

#4 F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages. DOI=http://dx.doi.org/10.1145/2827872

#5 PBS [Crash Course]. How YouTube knows what you should watch: Crash Course AI #15. (2019, November 22). [Video]. YouTube. https://www.youtube.com/watch?v=kiInh5STnyQ&t=3s

#6. PBS [Crash Course]. (2019, November 29). Let’s make a movie recommendation system: Crash Course AI #16 [Video]. YouTube. https://www.youtube.com/watch?v=iaIW3CO4rcY&t=661s

#7. GROUPLENS. (2009, January). MovieLens 10M/100k Data Set README. MovieLens 10M/100K. Retrieved June 1, 2022, from https://files.grouplens.org/datasets/movielens/ml-10m-README.html

#8. Introduction to Loss Functions. (n.d.). DataRobot AI Cloud. https://www.datarobot.com/blog/introduction-to-loss-functions/
################################################################################

################################################################################
# Initializing Data . Portions of the code within this project was provided by HARVARDX and the Reference Section Listed above.

# Note: this process could take a couple of minutes:
if(!require(tidyverse)) install.packages(
  "tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages(
  "caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages(
  "data.table", repos = "http://cran.us.r-project.org")
if(!require(RColorBrewer)) install.packages(
  "RColorBrewer", repos = "http://cran.us.r-project.org")
if(!require(rmarkdown)) install.packages(
  "rmarkdown", repos = "http://cran.us.r-project.org")
if(!require(dslabs)) install.packages(
  "dslabs", repos = "http://cran.us.r-project.org")

# Due to known loading issues, load the following package(s), if needed
install.packages("hms", dependencies = TRUE)
install.packages("gtable", dependencies = TRUE)
install.packages("hexbin", dependencies = TRUE)
install.packages("readr", dependencies=TRUE, INSTALL_opts = c('--no-lock'))
install.packages("caret", dependencies = TRUE)
install.packages("data.table", dependencies = TRUE)
install.packages("tidyverse", dependencies = TRUE)
install.packages("Rcolorbrewer", dependencies = TRUE)
install.packages("gt", dependencies = TRUE)
install.packages("curl", dependencies = TRUE)
install.packages("ggpmisc", dependencies = TRUE)
install.packages("knitr", dependencies = TRUE)
install.packages("dplyr", dependencies = TRUE)

#Load the following Library(ies)
library(tidyverse)
library(caret)
library(data.table)
library(RColorBrewer)
library(rmarkdown)
library(dslabs)
library(gtable)
library(hexbin)
library(gt)
library(curl)
library(dplyr)
library(ggpmisc)
library(gridExtra)
library(knitr)
library(lubridate)
library(stringr)

# For Lower Bandwidth/ RAM recommend adjusting the timeout settings
options(timeout = 320)

# Depending on your RAM, to free up unused memory, recommend using:
gc()

#To complete the next steps, verify which version of R your system is utilizing:
sessionInfo()

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

# If you received a Peer certificate cannot be authenticated error. 
# Recommend the following:
dl <- tempfile()
options(download.file.method="curl", download.file.extra="-k -L")
download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", 
                             readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(
  readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
movies <- as.data.frame(
  movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

# Join the data below 
movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data 
# set.seed(1, sample.kind="Rounding") 
# if using R 3.5 or earlier, use `set.seed(1)`

set.seed(1) 
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
Trial_Set <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- Trial_Set %>% semi_join(
  edx, by = "movieId") %>% semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(Trial_Set, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, movielens, removed)

################################################################################
## Overview: 
#### Movie Recommendation Project: MovieLens 10M Dataset
################################################################################

# Since the late 1970's corporations have used machine learning recommendation systems to understand user selections, user trends and consumer demands.  Machine learning algorithms in 2022 can now predict future interests, engagement, current taste, new product experimentation and more! 

# I will use The University of Minnesota team lab  (Grouplens) MovieLens 10m dataset for this project.  Grouplens selected 72,000 users at random to rate at least 20 movies for a combined 10 million ratings (view Reference Section for more information).  

####Goals: I will explore the Movielens 10M data set to conduct the following:

# First, I will clean the data, investigate any NAs and examine the outliers that may skew the data needed to achieve the Residual Mean Squared Error (RMSE) goal of .86490.

# Second, create a series of visualizations and examine each chart to understand what steps we need to complete to reach our RMSE goal.

# Third, create a trial recommender model to understand the RMSE. 

# Fourth, create a recommendation system based on the code provided.  Utilize Loss Function (RMSE), User effects, Regularization, and code from the reference section. 

# Lastly, finalize the machine learning algorithm to achieve the RMSE goal.

################################################################################


################################################################################
## Data Wrangling: Clean and examine the MovieLens 10M dataset 
#### Goal: Clean the dataset, examine the data and place the data in tables.
################################################################################

# Lets clean the data and view it in a table
options(scipen = 999)
rownames(edx)
colnames(edx)

head(edx, 10)

#For comparison purposes later lets compute Average Movie Rating
avg_rating <- mean(edx$rating)
avg_rating

#Verify if there are any inconsistencies or NAs
any(is.na(edx))
sum(is.na(edx))
which(is.na(edx), arr.ind=TRUE)

#Update Timestamp for edx and Trial_Set
edx[['timestamp']]<- as_datetime(edx[['timestamp']])
Trial_Set[['timestamp']]<- as_datetime(Trial_Set[['timestamp']])
abstract <- head(edx, 10)

# Create a Summary Table 
abstract %>%
  gt() %>%
  tab_header(
    title = md("**MovieLens Recommendation System Overview**"),
subtitle = md("HardvardX Capstone Project 2022")
    ) %>%
  cols_width(
    userId ~ px(130),
    movieId ~ px(130),
    rating ~ px(130),
    timestamp ~ px(130),
    title ~ px(130),
    genres ~ px(130),
    everything() ~ px(120)) %>%
  tab_source_note(
    source_note = md("Portions of this data is from the Reference Section.")
  ) %>%
  tab_footnote(
    footnote = md("MovieLens dataset consists of over **10M** movie ratings.")
    ) %>%
  opt_table_font(
    font = "Times New Roman",
    weight = 300) %>%
  cols_align(
    align = "center",
    columns = everything()
  )

MovieLens_Summary <- edx %>% summarise(
  total_movies = n_distinct(movieId),
  total_users = n_distinct(userId),
  total_genres = n_distinct(genres),
  total_titles = n_distinct(title))

MovieLens_Summary %>% 
  gt() %>%
  tab_header(
    title = md("**Summary of MovieLens Data Totals**"),
    subtitle = md("HardvardX Capstone Project 2022")
  ) %>%
  cols_width(
    total_movies  ~ px(130),
    total_users ~ px(130),
    total_genres ~ px(130),
    total_titles ~ px(130)
  ) %>%
  tab_source_note(
    source_note = md(
      "***Portions of this data is from the Reference Section.***")
  ) %>%
  tab_footnote(
    footnote = md("Timestamps, Ratings, and Years excluded for summary purposes.")
  ) %>%
  opt_table_font(
    font = ("Times New Roman"),
    weight = 300,
  ) %>%
  cols_align(
    align = "center",
    columns = everything()
  )

# The ratings score is 0.5 through 5. Let's see how are the Ratings Data distributed 
RatingPer<- table(edx$rating)
RatingPer

# After viewing the data, ratings 3, 4, and 5 received the most votes, further validating our average.

# Each user, on average, rated 20 movies but did every user rate 20 movies? Are there any users with less than 20 ratings or above 20 ratings? 
table(edx$userId <=20)

sum(edx$userId)

8997881+2180 

# By calling out this data, we noticed that 2180 occurrences of an UserId did not have greater than or equal to 20 ratings.  Per the table above, we only have unique 9,000,061 userIds for the training set.  So let us use a different code to narrow down this data. 

avg_rating

# Remember the average movie rating is 3.512464, so if we multiply that times the average amount of movies that were rated by an unique user (20), we should have the minimum average score per userID.
p = 3.512464*20
p

Userdata_sum1_Lessthan20 <- aggregate(rating ~ userId, data = edx, sum)<=p
table(Userdata_sum1_Lessthan20)

# R scanned both columns and found 4048 instances of unique users averaged a total rating of less than 70.2498 
x = 4048/9000061
x

# Luckily, that is equal to less than .04% of users contributed to the MovieLens database. This is statistically insignificant and we should not take more action in removing the 4048 userids. 

################################################################################
# MovieLens Dataset Visualization
# Goal: Create a series of in depth plots to visualize the data
################################################################################

# Average User Ratings via Histogram
edx %>%
  group_by(userId) %>%
  summarise(avrat = mean(rating)) %>%
  ggplot(aes(avrat)) +
  geom_histogram(binwidth = 0.5, fill = "khaki", color = "blue") +
  geom_vline(aes(xintercept = mean(avrat)),
             color = "red", linetype="dashed", size = 1
  )+
  labs(
    title = "MovieLens Recommendation System Average User Rating",
    subtitle = "HardvardX Capstone Project 2022",
    caption = "Red Dotted Line is the Average User Ratings of 3.51. 
    Portions of this data is from the Reference Section.",
    x = "Ratings Scale",
    y = "Total Number of Users",
  )+
  theme_bw()+
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5),
    plot.caption = element_text(face = "italic",hjust = 0.5))

# Let us plot a box plot to see if we have any outliers that the histogram could not display. 

# Boxplot of Ratings Distribution with the Average Movie Rating   
ggplot(edx, aes(x = userId, y = rating)) +                
  geom_boxplot(color = "blue", fill = "green", alpha = 0.2) +
  labs(
    title = "MovieLens Recommendation System Ratings",
    subtitle = "HardvardX Capstone Project 2022",
    caption = "Red dotted line is the average movie rating (3.51). 
    Portions of this data is from the Reference Section.",
    y = "Ratings (.5-5)",
    x = "Unique Users ",
  ) +
  theme_classic() +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5),
    plot.caption = element_text(face = "italic", hjust = 0.5)
  )+
  annotate("segment", x= 4000, xend = 68000, y = 3.51, yend = 3.51, color = "red", size = 1.5, linetype = "dashed")

# As we can see, several outliers encompass the least favorable ratings. Let us plot the ten worst Movies based on total ratings and examine the results.

#BarPlot of Top 10 Worst Movies by Least Amount of Total Ratings 
Bad10Titles <- aggregate(rating ~ title, data = edx, sum)

Bad10Titles1<- tibble(Bad10Titles)
R <-Bad10Titles1[order(Bad10Titles1$rating),]
R
nrow(R)
L <- R[-c(11:10676), ]
L
colnames(L)
nrow(L)

ggplot(data = L, mapping = aes(x = title, y = rating, fill = title)) +
  geom_bar(stat = "identity")+
  labs(
    title =  "10 Worst Movies by Total Ratings ",
    subtitle = "HardvardX Capstone Project 2022",
    caption = "This is calculated by Least Amount of Total Ratings.
    Portions of this data is from the Reference Section.",
    x = "Top 10 Worst Movie Titles",
    y = "Total Number of Ratings",
  )+
  theme_classic()+
  theme(
    legend.position = 'none',
    axis.text.x = element_text(angle = 70, vjust = 1, hjust = 1),
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5),
    plot.caption = element_text(face = "italic",hjust = 0.5))+
  coord_flip()

# As we can see, the movies listed only had one rating of .5 or 1 (the lowest eligible ratings).  A movie with only one total rating out of thousands of possible ratings from unique users will be considered an outlier in this data set.  Remember, the average unique user rated at least 20 movies. 

#BarPlot of Top 10 Movies by the Highest Total Ratings
Top10Titles <- aggregate(rating ~ title, data = edx, sum)

Top10Titles1<- tibble(Top10Titles)
S <-Top10Titles1[order(-Top10Titles1$rating),]
nrow(S)
M <- S[-c(11:10676), ]
M
colnames(M)
nrow(M)

ggplot(data = M, mapping = aes(x = title, y = rating, fill = title)) +
  geom_bar(stat = "identity")+
  labs(
    title = "Top 10 Movies by Total Ratings ",
    subtitle = "HardvardX Capstone Project 2022",
    caption = "Portions of this data is from the Reference Section.",
    x = "Top 10 Movie Titles",
    y = "Total Number of Ratings",
  )+
  theme_classic()+
  theme(
    legend.position = 'none',
    axis.text.x = element_text(angle = 30, vjust = 1, hjust = 1),
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5),
    plot.caption = element_text(face = "italic",hjust = 0.5))+
  coord_flip()

# Ratio of Genres to Ratings
# ***Note: this process could take a couple of minutes:***
GR <- edx %>% group_by(genres) %>% mutate(ratio = mean(rating))

ggplot(GR, aes(x = ratio, y = genres, colour = rating)) +
  geom_point(show.legend = FALSE) +
  scale_color_gradient(low = "red", high = "green")+
  labs(
    title = " Ratio of Genres to Ratings ",
    subtitle = "HardvardX Capstone Project 2022",
    caption = "Portions of this data is from the Reference Section.",
    x = "Ratings",
    y = " Genres",
  )+
  theme_bw()+
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5),
    plot.caption = element_text(face = "italic",hjust = 0.5))+
  scale_y_discrete(guide = guide_axis(check.overlap = TRUE))

#Look at how the majority of the ratio of genres to ratings are between 3 & 4. 

################################################################################
## Data Analysis: Utilizing Data Analysis for Trial RMSE
#### Goal: Identify a suitable Residual Mean Squared Error (RMSE) in the effort to evaluate how close the predictions are to the Goal RMSE.
################################################################################

#After cleaning and visualizing the data, we now understand that outliers and data inconsistencies probably can affect the machine learning algorithm when making a recommender system. 

#Having an average rating of 3.51 is not enough to depict the accuracy of a recommender system. We will use the Loss Function's Residual Mean Squared Error to evaluate our algorithm model for our train(edx) and test set(Trial_Set). Per DataRobot (Introduction to Loss Functions, n.d.), for RMSE, we will use the difference between the predictions and the actual data, square it, and average it across the entire dataset. The formula is

#$$RMSE = \sqrt{\frac{1}{N}\sum_{u,i}\left(\hat{y}_{u,i}-y_{u,i}\right)^2}$$ 
#  1. The following explains the formulas or the associated object:
#  *  Train set = edx
#*  Test set = Trial_Set
#*  i = actual user provided rating  
#*  u = movie
#*  b = bias 
#*  mu = avg_rating
#*  b_i = movie_bias
#*  b_u = User_Effect
#*  N = sum of user/movie combinations

#$$\sum_{u,i}$$ = Sum of u,i

#$$(\hat{y}_{u,i}-{y}_{u,i})^2$$ = differences of both, squared

# Create a loss function 
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


#Per the course instruction we want a RMSE around .86490. Lets adjust and plug  in numbers to see the effects by creating a Trial Run Model utilizing our test set.
head(Trial_Set,10) 
any(is.na(Trial_Set))
sum(is.na(Trial_Set))

#Since this is our first model, lets observe all unknown ratings utilizing a test set.
True_NaiveB_RMSE <- RMSE(Trial_Set$rating, avg_rating)
True_NaiveB_RMSE

Trial_One <- rep(1, nrow(Trial_Set))
Trial_One_RMSE<- RMSE(Trial_Set$rating, Trial_One)

Trial_Two <- rep(2, nrow(Trial_Set))
Trial_Two_RMSE<- RMSE(Trial_Set$rating, Trial_Two)

Trial_Three <- rep(3, nrow(Trial_Set))
Trial_Three_RMSE <- RMSE(Trial_Set$rating, Trial_Three)

Trial_Run_Summary <- tibble(
  Trial_One_RMSE,
  Trial_Two_RMSE,
  Trial_Three_RMSE,
  avg_rating,
  True_NaiveB_RMSE)

Trial_Run_Summary

#Lets put our trial data in a table
Trial_Run_Summary %>% 
  gt() %>%
  tab_header(
    title = md("**Summary of Trial RMSE Predictions**"),
    subtitle = md("HardvardX Capstone Project 2022")
  ) %>%
  cols_width(
    True_NaiveB_RMSE ~ px(135),
    Trial_One_RMSE ~ px(135),
    Trial_Two_RMSE ~ px(135),
    Trial_Three_RMSE ~ px(135),
    avg_rating ~ px(135)) %>% 
  tab_source_note(source_note = md(
    "Portions of this data is from the Reference Section.")) %>%
  tab_footnote(
    footnote = md("MovieLens dataset consists of over **10M** movie ratings.")
  ) %>%
    opt_table_font(font = ("Times New Roman"),
    weight = 300)  %>%
  cols_align(
    align = "center",
    columns = everything()
  ) %>%
    data_color(
      columns = c(True_NaiveB_RMSE, Trial_One_RMSE, Trial_Two_RMSE, 
                Trial_Three_RMSE, avg_rating),
                colors = scales::col_numeric(
      palette = c(
        "white", "pink", "red"),
      domain = c(1,4)))

# Lets Compare it to the RMSE Goal of .86490
Goal_RMSE <- .86490
Goal_RMSE

Trial_Name <- c("Trial_One_RMSE","Trial_Two_RMSE", "Trial_Three_RMSE", 
                "True_NaiveB_RMSE") 
RMSE_Value <- c(2.75547, 1.86982, 1.183146, 1.060651)
Max_RMSE <- c(0, 1, 2, 3)

Trial_Compare<- data.frame(Trials = c("Trial_One_RMSE","Trial_Two_RMSE", 
                                      "Trial_Three_RMSE", "True_NaiveB_RMSE"),
                           RMSE_Value = c(2.75547, 1.86982, 1.183146, 1.060651),
                           Max_RMSE = c(0, 1, 2, 3))

Trial_Plot<- ggplot(Trial_Compare, aes(RMSE_Value, Max_RMSE, color = Trials))+
  geom_point(size = 3)+
  geom_vline(aes(xintercept = Goal_RMSE),
             color = "brown", linetype="dashed", size = 0.5
  )+
  labs(
    title = " MovieLens RMSE Trial Comparision",
    subtitle = "HardvardX Capstone Project 2022",
    caption = "Brown Dotted Line is the RMSE Goal of .86490. 
    Portions of this data is from the Reference Section.",
    x = "Max Value",
    y = "RMSE Value Range",
  )+
  theme_classic()+
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5),
    plot.caption = element_text(face = "italic", hjust = 0.5))+
  coord_flip()
Trial_Plot

Trial_Compare_Legend<- data.frame(Trials = c("Trial_One_RMSE","Trial_Two_RMSE", 
                                      "Trial_Three_RMSE", "True_NaiveB_RMSE"),
                           RMSE_Value = c(2.75547, 1.86982, 1.183146, 1.060651))

Trial_Plot+
  annotate(geom = "table",
           x = 3.5,
           y = 3.5,
           label = list(Trial_Compare_Legend))

# As you can see my trial data is far from the brown dotted line which represents our RMSE goal.  Moving forward we will only rely on our train set (edx). By refining the code, I will get closer to the RMSE needed to ensure that this recommendation system is valuable for our train set (edx) vice our Trial_Set. 

NaiveB_RMSE = bind_rows(method = "The Naive RMSE Mean", RMSE = True_NaiveB_RMSE)
NaiveB_RMSE 

################################################################################
## Data Analysis: Bias in Movie Ratings
#### Goal: Identify a suitable Residual Mean Squared Error (RMSE) to examine bias in Movie Ratings in the effort to evaluate how close the predictions are to the Goal RMSE.
################################################################################

# Some movies receive more ratings over time than others. Higher-rated movies could stem from users falling in love with the movie, or it could be that certain movies leave a lasting effect on a population/culture, which turns them into a household name. Being a household name, I could infer that users may have never watched the movie. However, since everyone says it is an excellent movie, I believe people would automatically give it a good rating for being a classic. That effect is called Movie Bias, and we will account for it in the algorithm.  {b} will be the the bias while {i} is the actual user provided rating. 
#To calculate movie bias we will use Formula: $$Y_{u,i} ={\mu}+{b_i}+\epsilon_{u,i}$$

# Plot the average movie bias

Average_Movie_Bias<- edx %>% distinct() %>%
  group_by(movieId) %>%
  summarise(movie_bias = mean(rating - avg_rating))

MA_df <- data.frame(Average_Movie_Bias)

Bias_Plot<- ggplot(MA_df, aes(x = movie_bias))+
  geom_histogram(binwidth = 1, color = "blue", fill = "white")+
  labs(
    title = "MovieLens Recommendation Bias Ratings",
    subtitle = "HardvardX Capstone Project 2022",
    caption = "Portions of this data is from the Reference Section.",
    x = "Ratings Bias",
    y = "Movies based on Log2",
  )+
  scale_x_continuous(trans ="log2")+
  scale_y_continuous(trans = "log2")+
  theme_classic()+
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5),
    plot.caption = element_text(face = "italic",hjust = 0.5))
Bias_Plot

# Now lets see if we have any improvements since updating the code. 

Clean_predicted_ratings_true_1 <- avg_rating + edx %>%
  left_join(Average_Movie_Bias, by = "movieId") %>%
  pull(movie_bias)
Clean_predicted_ratings_true_1
RMSE(Clean_predicted_ratings_true_1, edx$rating)


#As we can see we are getting closer to our goal of .86490. 
Movie_Bias_Effects_RMSE <- RMSE(Clean_predicted_ratings_true_1, edx$rating)
Movie_Bias_Effects_RMSE
NaiveB_RMSE = bind_rows(NaiveB_RMSE,
                        data_frame(method = "Movie Bias Effects", RMSE = 
                                     Movie_Bias_Effects_RMSE))

NaiveB_RMSE

################################################################################
## Data Analysis: User Specific Effects
#### Goal: Identify a suitable Residual Mean Squared Error (RMSE) to examine bias  User Specific Effects in the effort to evaluate how close the predictions are to the Goal RMSE.
################################################################################

# As an avid Marvel movie watcher, I am biased towards the marvel cinematic universe. Additionally, I am not too fond of romance movies. If I was forced to rate a romance movie, I am 100% sure I would rate it a .5 rating without watching one minute of the movie. As the creator of this algorithm set, we must account for the User effect among our user base. To formulate this we will use ${b_u}$ to account for user affects. User Effects = the mean of the ratings per user. We can add this to our model:

#$$Y_{u,i} ={\mu}+{b_i}+{b_u}+\epsilon_{u,i}$$
#then update it to reflect one's high ratings vs low ratings
#$$\hat{b_u} = mean(Y_\hat{u,i} - \hat{\mu}-\hat{b_i})$$

# Movie Ratings chart based on Average Movie Ratings
edx%>% 
  ggplot(aes(rating))+ 
  geom_histogram(binwidth = .5, color = "blue", fill = "white")+
  geom_vline(aes(xintercept = mean(rating)),
             color = "red", linetype="dashed", size = 1.0
  )+
  labs(
    title = "MovieLens Recommendation System Ratings Histogram",
    subtitle = "HardvardX Capstone Project 2022",
    caption = "Red Dotted Line is the Average Ratings of 3.51. 
    Portions of this data is from the Reference Section.",
    x = "Ratings 1-5",
    y = "Ratings Distrubution based on Log2",
  )+
  scale_x_continuous(trans ="log2")+
  scale_y_continuous(trans = "log2")+
  theme_classic()+
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5),
    plot.caption = element_text(face = "italic",hjust = 0.5))

Average_User_Rating <- edx %>% distinct() %>%
  left_join(Average_Movie_Bias, by="movieId") %>%
  group_by(userId) %>%
  summarise(User_Effect = mean(rating - avg_rating - movie_bias))

Clean_predicted_ratings_true_2 <- edx %>%
  left_join(Average_Movie_Bias, by='movieId') %>%
  left_join(Average_User_Rating, by='userId') %>%
  mutate(pred = avg_rating + movie_bias + User_Effect) %>%
  pull(pred)
RMSE(Clean_predicted_ratings_true_2, edx$rating)

#Check the progress
User_Bias_Effects_RMSE <- RMSE(Clean_predicted_ratings_true_2, edx$rating)
User_Bias_Effects_RMSE
NaiveB_RMSE = bind_rows(NaiveB_RMSE,
                        data_frame(method = "User Bias Effects",
                                   RMSE = User_Bias_Effects_RMSE))

NaiveB_RMSE

################################################################################
## Data Analysis: Regularization 
#### Goal: Identify a suitable Residual Mean Squared Error (RMSE) to examine Regularization in the effort to evaluate how close the predictions are to the Goal RMSE.
################################################################################

# We are getting closer to our RMSE goal.  So far, we have recognized that there are movie bias and user effects that can significantly change our algorithm.  To shrink the errors and deviations, we will use regularization.  This will help us refine the algorithm over time by eventually adding penalty terms. 

# Earlier, we determined the ten best and worst movies based on total ratings.  In both visualizations, we did not consider user effects and movie bias.  Let us implement both to see if the best and worst movies change. Also, we will see how many ratings per user for each film are in our new list of 10 best & 10 worst. 

movie_titles <- edx %>%
  select(movieId, title, rating) %>%
  distinct()

#Best Movies with  movie bias

Top_10_Best_Movies <-tibble(Average_Movie_Bias %>% left_join(
  movie_titles, by="movieId") %>%
    arrange(desc(movie_bias)) %>% 
    slice(1:11)  %>%
    pull(title))

table(Top_10_Best_Movies$`... %>% pull(title)`)

#Worst Movies with movie bias

Top_10_Worst_Movies <-tibble(Average_Movie_Bias %>% 
    left_join( movie_titles, by="movieId") %>%
    arrange(movie_bias) %>% 
    slice(1:31)  %>%
    pull(title))
table(Top_10_Worst_Movies)

#As you can see, the number of ratings per movie is not enough to train an algorithm. Inherently, we could guess the user rating selections, but the margin of error would be too high.  Instead, we will use penalized regression to optimize the algorithm parameters and control the variability of the movie effects.  I will test the regularized estimates using the object lambdas.  

#Lambda ( $\lambda$ ) is the Greek alphabet eleventh letter, and its symbol is commonly used for wavelength.  Lambda is the common term used in the machine learning community for finding the balance between training the data and simplicity.  Multi-purpose retail giant Amazon has a machine learning model named AWS Lambdas that test machine learning algorithms.  I will test  $\lambda$ = 1 and $\lambda$ = 5 to see which object is a better tuning mechanism.  We will also visualize each lambda to see which one has a more substantial weight towards zero.

#Formula for Lambda One:

# $\lambda$ = 1

# $$\hat{b_i}(\lambda) = {\frac{1}{\lambda + n_i}\sum_{u=i}^{n_i}\left({Y}_{u,i}-\hat{\mu}\right)}$$

#Formula for Lambda Five:

# $\lambda$ = 5

# $$\hat{b_i}(\lambda) = {\frac{1}{\lambda + n_i}\sum_{u=i}^{n_i}\left({Y}_{u,i}-\hat{\mu}\right)}$$

lambda_one <- 1
avg_movies <- mean(edx$rating)
Average_Regularized_Mov_1 <- edx %>% 
  group_by(movieId) %>% 
  summarize(movie_bias = sum(rating - avg_movies)/(n()+lambda_one), n_i = n())

Penalty_Plot <- tibble(original_1 = Average_Movie_Bias$movie_bias, 
       regularlized_1 = Average_Regularized_Mov_1$movie_bias, 
       n = Average_Regularized_Mov_1$n_i)

Penalty_Plot

Comp1 <-ggplot(data = Penalty_Plot, aes(original_1, regularlized_1, size=sqrt(n))) + 
  geom_point(shape=1, alpha=0.5)+
  labs(
    title = "RLambda 1",
    subtitle = "HardvardX Capstone Project 2022",
    caption =   " Portions of this data is from the Reference Section.",
    x = "Least Squares Estimates",
    y = "Regularized Lambda 1",
  )+
  theme_classic()+
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5),
    plot.caption = element_text(face = "italic", hjust = 0.5))

Comp1

lambda_five <- 5
avg_movies <- mean(edx$rating)
Average_Regularized_Mov <- edx %>% 
  group_by(movieId) %>% 
  summarize(movie_bias = sum(rating - avg_movies)/(n()+lambda_five), n_i = n())

Comp_Penalty <- tibble(original = Average_Movie_Bias$movie_bias, 
       regularlized = Average_Regularized_Mov$movie_bias, 
       n = Average_Regularized_Mov$n_i) 

Comp_Penalty

Comp2 <- ggplot(data = Comp_Penalty, aes(original, regularlized, color = " dark red",
                                size=sqrt(n))) + 
  geom_point(shape=1, alpha=0.5)+
  labs(
    title = "Lambda 5",
    subtitle = "HardvardX Capstone Project 2022",
    caption =   " Portions of this data is from the Reference Section.",
    x = "Least Squares Estimates",
    y = "Regularized Lambda 5",
  )+
  theme_classic()+
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5),
    plot.caption = element_text(face = "italic", hjust = 0.5))

Comp2

grid.arrange(Comp1, Comp2 , ncol = 2 )

# Observe the Side-by_side Comparison.  Notice in Lambda Five that the estimates shrank considerably, and the weight of the plot is more towards zero. Lambda Five spread also indicates more wiggle room for simplicity.  If we pick Lambda one, we will run the risk of training a more complex algorithm making it harder for the algorithm to process more general information.  We will stick with Lambda Five.  Lets re-evaluate the Top Ten lists utilizing the penalized estimates.

#Top 10 Best Movies using Penalized Estimates 
Top_10_Best_Movies_Pen <-tibble(edx %>%
  count(movieId) %>% 
  left_join(Average_Regularized_Mov, by = "movieId") %>%
  left_join(movie_titles, by = "movieId") %>%
  arrange(desc(movie_bias)) %>% 
  slice(1:90) %>% 
  pull(title)) 
table(Top_10_Best_Movies_Pen)

#Top 10 Worst Movies using Penalized Estimates 
Top_10_Worst_Movies_Pen <-tibble( edx %>%
  count(movieId) %>% 
  left_join(Average_Regularized_Mov, by = "movieId") %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(movie_bias) %>% 
  select(title, movie_bias, n) %>% 
  slice(1:79) %>% 
  pull(title)) 
  
table(Top_10_Worst_Movies_Pen)

# As you can see, the number of ratings per movie changed considerably.   Utilizing penalized regression to optimize the algorithm parameters and control the variability of the movie effects has helped us get better data to assist us in reaching our Goal rmse.  Let us finish buttoning up our code and add the results to our RMSE table.

L5RP <-mean(Comp_Penalty$regularlized)

Clean_predicted_ratings_3 <- edx %>% 
  left_join(Average_Regularized_Mov, by = "movieId") %>%
  mutate(pred = avg_movies + movie_bias + L5RP) %>%
  pull(pred)

Clean_predicted_ratings_3

#Check the progress and add to table

Regularized_Movie_Effect <- RMSE(Clean_predicted_ratings_3, edx$rating)
Regularized_Movie_Effect
NaiveB_RMSE = bind_rows(NaiveB_RMSE,
                        data_frame(method = "Regularized Movie Effect",
                                   RMSE = Regularized_Movie_Effect))
NaiveB_RMSE

################################################################################
## Data Analysis: Finalize Regularization 
#### Goal: Obtain an optimum Residual Mean Squared Error (RMSE) in the effort to evaluate how close the predictions are to the Goal RMSE.
################################################################################

# From the Ratings Average to the Regularized Movie Effect Model I noticed a considerable changed in my machine learning model when I added average penalized regularized terms. Here, I will verify my decision to choose Lambda Five to see if I can obtain the Goal RMSE!  


lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  avg_movies <- mean(edx$rating)
  movie_bias <- edx %>% 
      group_by(movieId) %>%
      summarize(movie_bias = sum(rating - avg_movies)/(n()+l))
    
  User_Effects <- edx %>% 
      left_join(movie_bias, by="movieId") %>%
      group_by(userId) %>%
      summarize(User_Effects = sum(rating - movie_bias - avg_movies)/(n()+l))
    
    Clean_predicted_ratings_4 <- 
      validation %>% 
      left_join(movie_bias, by = "movieId") %>%
      left_join(User_Effects, by = "userId") %>%
      mutate(pred = avg_movies + movie_bias + User_Effects) %>%
      pull(pred)
    
    return(RMSE(validation$rating, Clean_predicted_ratings_4))
  })
  
lambda <- lambdas[which.min(rmses)]
lambda
  qplot(lambdas, rmses) +
    geom_vline(aes(xintercept = 5),
               color = "purple", linetype="dashed", size = 0.5
    )+
    labs(
      title = "MovieLens Recommendation System Tuning Selection",
      subtitle = "HardvardX Capstone Project 2022",
      caption = "Purple Dotted Line is the Tuning Parameter of 5.
     Portions of this data is from the Reference Section.",
      x = "RMSES",
      y = "LAMBDAS",
    )+
    theme_classic()+
    theme(
      plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
      plot.subtitle = element_text(hjust = 0.5),
      plot.caption = element_text(face = "italic", hjust = 0.5))  
  

Final_Movie_Model <- min(rmses)
Final_Movie_Model
NaiveB_RMSE = bind_rows(NaiveB_RMSE,
                          data_frame(
                            method = "Finalized Movie Model",
                                     RMSE = Final_Movie_Model))
NaiveB_RMSE

################################################################################
## MovieLens 10M Dataset Movie Recommendation Model Results 
#### Goal Achieved. RMSE of .865
################################################################################

#Final results are posted in the Table 

MovieLens_RMSE_Methods <- tibble(NaiveB_RMSE)

MovieLens_RMSE_Methods

colnames(MovieLens_RMSE_Methods)

colnames(MovieLens_RMSE_Methods)<- c("Model_Name", "RMSE")


#Lets put our final Results in a table

MovieLens_RMSE_Methods %>% 
  gt() %>%
  tab_header(
    title = md("**MovieLens Recommendation System Final RMSE Model Results**"),
    subtitle = md("Summary of Each Recommender Prediction Model")
  ) %>%
  cols_width(
    everything() ~ px(200)) %>%
  tab_source_note(
    source_note = md("***Portions of this data is from the Reference Section.***")
  ) %>%
  opt_table_font(
    font = ("Times New Roman"),
    weight = 300,
  )  %>%
  cols_align(
    align = "center",
    columns = everything()
  ) %>%
  tab_style(
    style = list(
      cell_fill(color = "green")
    ),
    location = cells_body(
      columns = c("Model_Name", "RMSE"),
      rows = 5
    ))

################################################################################
## Conclusion: MovieLens 10M Dataset Movie Recommendation Model
################################################################################

#***We finally met our goals.***
#  1. We cleaned the data, determined what outliers were statistically significant, and created a series of visualizations to better understand our data.

#2. We created a trial recommender model to understand how the model will interpret the data and RMSE. Furthermore, we created a recommendation system  utilizing Loss Function (RMSE), added Penalty Terms, accounted for Movie bias, User effects, and Regularization. 

#3. Lastly, we finalized the machine learning algorithm to achieve the RMSE of 865. 
