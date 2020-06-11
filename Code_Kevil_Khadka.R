# Title: "A Survey on Machine Learning Techniques for Airbnb Price Prediction"
# Author: "Kevil Khadka"
# Date: "4/10/2020"

  
## Loading libraries
library(tidyverse)
library(dplyr)
library(caret)
library(MASS)
library(Sleuth3)
library(ISLR)
library(mice)
library(Hmisc)
library(GGally)
library(leaps)
library(caTools)
library(ggthemes)
library(ggExtra)
library(glmnet)
library(corrplot)
library(leaflet)
library(kableExtra)
library(RColorBrewer)
library(plotly)
library(plotrix)
library(readr)
library(ggmap)
library(ggpubr)

# Decision Tree
library(rpart)
library(rpart.plot)
library(randomForest)
library(partykit)

# Naive Bayes
library(e1071)
library(klaR)

# kNN
library(class)

# Text Analaysis
library(tidytext)
library(wordcloud)

# Sentiment Analysis
library(textdata)
library(reshape2)

# File Location
setwd("/System/Volumes/Data/University of Evansville/SPRING 2020/STAT 493/Airbnb_final project/RCode/Final_Coding_Khadka_files/Datasets")

## Importing dataset
# Los Angeles
set.seed(123)
losangeles <- read_csv("/System/Volumes/Data/University of Evansville/SPRING 2020/STAT 493/Airbnb_final project/RCode/Final_Coding_Khadka_files/Datasets/LA_listingsDec18.csv")
summary(losangeles)


## Data Exploration
airbnb_LA <- dplyr::select(losangeles, -id, -last_review)
airbnb_LA$neighbourhood_group <- as_factor(airbnb_LA$neighbourhood_group)
airbnb_LA$neighbourhood <- as_factor(airbnb_LA$neighbourhood)
airbnb_LA$room_type <- as_factor(airbnb_LA$room_type)
summary(airbnb_LA)


# Missing Data
contents(airbnb_LA)

missing_airbnb <- airbnb_LA %>% summarise_all(~(sum(is.na(.))/n()))
missing_airbnb <- gather(missing_airbnb, key = "variables", value = "percent_missing")
missing_airbnb <- missing_airbnb[missing_airbnb$percent_missing > 0.0, ]

th <- theme_fivethirtyeight() + theme(axis.title = element_text(), axis.title.x = element_text())

ggplot(missing_airbnb, aes(x = reorder(variables, percent_missing), y = percent_missing)) + 
  geom_bar(stat = "identity", fill = "red", aes(color = I('white')), size = 0.3) +
  xlab('variables') +
  coord_flip() +
  th +
  ggtitle("Missing Data") +
  xlab("Column name") +
  ylab("Percentage missing") +
  annotate("text", x = 1.5, y = 0.1, label = "name and host_name has less than 0.001\n percentage missing", color = "slateblue", size = 5)


# Data cleaning
# Set NA to value '0'
airbnb_LA$reviews_per_month[is.na(airbnb_LA$reviews_per_month)] <- 0
airbnb_LA$calculated_host_listings_count[is.na(airbnb_LA$calculated_host_listings_count)] <- 0
airbnb_LA$name[is.na(airbnb_LA$name)] <- "No.name"
airbnb_LA$host_name[is.na(airbnb_LA$host_name)] <- "No.name"

summary(airbnb_LA)


## Part 1: Data Visualization

# Leaflet map
pal <- colorFactor(palette = c("orange", "red", "green", "purple", "blue"), domain = airbnb_LA$neighbourhood_group)

leaflet(data = airbnb_LA) %>% 
  addProviderTiles(providers$CartoDB.DarkMatterNoLabels) %>% 
  addCircleMarkers(~longitude, ~latitude, color = ~pal(neighbourhood_group),
                   weight = 1, radius=1, fillOpacity = 0.1, opacity = 0.1,
                   label = paste("Name:", airbnb_LA$name)) %>% 
  addLegend("bottomright", pal = pal, values = ~neighbourhood_group,
            title = "Neighbourhood groups",opacity = 1)

# Looking Correlation
airbnb.LA.1 <- airbnb_LA %>%  
  select("price", "number_of_reviews", "minimum_nights", "availability_365", "neighbourhood_group", "room_type")

ggpairs(airbnb.LA.1, progress = FALSE)

# From following ggpairs graph, we cannot see much correlation between those variables.

# Looking our response variable: "price"
# Do not need the listing which has price value 0
airbnb_LA <- airbnb_LA %>%
  filter(price > 0) %>% 
  filter(reviews_per_month > 0) %>% 
  filter(number_of_reviews > 0) %>%
  filter(calculated_host_listings_count > 0)

# Histogram and Density
ggplot(airbnb_LA, aes(price)) +
  geom_histogram(bins = 30, aes(y = ..density..), fill = "red") + 
  geom_density(alpha = 0.2, fill = "red") +
  ggtitle("Distribution of price - Airbnb LA",
          subtitle = "The distribution is very skewed") +
  theme(axis.title = element_text(), axis.title.x = element_text()) +
  geom_vline(xintercept = round(mean(airbnb_LA$price), 3), size = 2, linetype = 3)

price_list <- airbnb_LA %>% 
  select(price) %>% 
  dplyr::group_by(price) %>% 
  dplyr::summarise(n_count = n()) %>% 
  dplyr::mutate(n_count_per = n_count * 100 / sum(n_count)) %>% 
  arrange(desc(n_count)) %>% 
  top_n(10)
print(price_list)

# Log Transform of "price"
airbnb_LA <- airbnb_LA %>%
  mutate(log_price = log(price))

ggplot(airbnb_LA, aes(log_price)) +
  geom_histogram(bins = 30, aes(y = ..density..), fill = "red") + 
  th +
  geom_density(alpha = 0.2, fill = "red") +
  ggtitle("Distribution of price - Airbnb LA",
          subtitle = "The distribution is normal") +
  theme(axis.title = element_text(), axis.title.x = element_text()) +
  geom_vline(xintercept = round(mean(airbnb_LA$log_price), 2), size = 2, linetype = 3)

# NOTE: 0.1 unit change in log(x) is equivalent to 10% increase in X.

# Log_Price vs Minimum Nights 
rew1 <- ggplot(airbnb_LA, aes(minimum_nights, log_price)) +
  th + 
  theme(axis.title = element_text(), axis.title.x = element_text()) +
  geom_point(aes(color = room_type), alpha = 1) +
  geom_smooth(method = "lm") +
  xlab("Minimum nights") +
  ylab("Log Price") +
  ggtitle("Relationship between price vs minimum nights")

nightstay_list <- airbnb_LA %>% 
  select(minimum_nights) %>% 
  dplyr::group_by(minimum_nights) %>% 
  dplyr::summarise(n_count = n()) %>% 
  dplyr::mutate(n_count_per = n_count * 100 / sum(n_count)) %>% 
  arrange(desc(minimum_nights))
print(nightstay_list)

fit4 <- lm(log_price ~ minimum_nights, data = airbnb_LA)
summary(fit4)

airbnb_LA.2 <- airbnb_LA %>% filter(minimum_nights <= 31)
# summary(airbnb_LA.2)

fit4.1 <- lm(log_price ~ minimum_nights, data = airbnb_LA.2)

fit4.2 <- lm(log_price ~ sqrt(minimum_nights), data = airbnb_LA)

summary(fit4)
summary(fit4.1)
summary(fit4.2)

hist(fit4.1$residuals)
hist(fit4.2$residuals)

# Square root Transform of "minimum_nights"
airbnb_LA <- airbnb_LA.2 %>%
  mutate(sqrt.min.nights = sqrt(minimum_nights))
# summary(airbnb_LA)

# Log_Price vs sqrt.Minimum Nights 
rew <- ggplot(airbnb_LA, aes(sqrt.min.nights, log_price)) +
  th + 
  theme(axis.title = element_text(), axis.title.x = element_text()) +
  geom_point(aes(color = room_type), alpha = 1) +
  geom_smooth(method = "lm") +
  xlab("Sqrt of Minimum nights") +
  ylab("Log Price") +
  ggtitle("Relationship between price vs minimum nights")

ggarrange(rew1, rew, nrow = 2, ncol = 1)


# Price vs Host_listings_count
ggplot(airbnb_LA, aes(calculated_host_listings_count, log_price)) +
  th +
  geom_point(alpha = 0.5, aes(color = neighbourhood_group)) +
  geom_smooth(method = "lm") +
  xlab("calculated_host_listings_count") +
  ylab("Log Price") +
  ggtitle("Relationship between price and calculated_host_listings_count") 

# Need to filter calculated_host_listings_count
hostlist_list <- airbnb_LA %>% 
  select(calculated_host_listings_count) %>% 
  dplyr::group_by(calculated_host_listings_count) %>% 
  dplyr::summarise(n_count = n()) %>% 
  dplyr::mutate(n_count_per = n_count * 100 / sum(n_count)) %>% 
  arrange(desc(calculated_host_listings_count)) 
print(hostlist_list)

# View host_listing more than 152
airbnb_LA %>% filter(calculated_host_listings_count == 152)


# Host Name: Oranj Palm & Catalina Island
# Listed Airbnb Location: Other Cities and Unincorporated Areas
# so, we can delete the data because it does not belong to Airbnb. 


fit7 <- lm(log_price ~ calculated_host_listings_count , data = airbnb_LA)
summary(fit7)
hist(fit7$residuals)

# Remove the list from the data
airbnb_LA <- airbnb_LA %>% 
  filter(calculated_host_listings_count != 152)
summary(airbnb_LA)

fit7.1 <- lm(log_price ~ calculated_host_listings_count , data = airbnb_LA)
summary(fit7.1)
hist(fit7.1$residuals)


# Price vs Availability
ggplot(airbnb_LA, aes(availability_365, log_price)) +
  th +
  geom_point(alpha = 0.5, aes(color = neighbourhood_group)) +
  geom_smooth(method = "lm") +
  xlab("Availability during year") +
  ylab("Log Price") +
  ggtitle("Relationship between price and availability")

available_list <- airbnb_LA %>% 
  select(availability_365) %>% 
  dplyr::group_by(availability_365) %>% 
  dplyr::summarise(n_count = n()) %>% 
  dplyr::mutate(n_count_per = n_count * 100 / sum(n_count)) %>% 
  arrange(desc(n_count)) 
print(available_list)

airbnb_LA %>% filter(availability_365 == 0)
# most busiest airbnb hosts seem fully reserved earlier
# Location: mostly City of LA

airbnb_LA <- airbnb_LA %>% 
  filter(availability_365>0)
summary(airbnb_LA)

fit8 <- lm(log_price ~ availability_365 , data = airbnb_LA)
summary(fit8)
hist(fit8$residuals)

ggplot(airbnb_LA, aes(availability_365, log_price)) +
  th +
  geom_point(alpha = 0.5, aes(color = neighbourhood_group)) +
  geom_smooth(method = "lm") +
  xlab("Availability during year") +
  ylab("Log Price") +
  ggtitle("Relationship between price and availability")


# Split 'price' into 3 parts (Low, Medium, High)
data_temp <- sort.int(airbnb_LA$price, decreasing = FALSE)
group_1 <- data_temp[round(length(data_temp)/3, digits = 0)]
group_2 <- data_temp[2*round(length(data_temp)/3, digits = 0)]
airbnb_LA$price_group[airbnb_LA$price <= group_1] <- "Low"
airbnb_LA$price_group[airbnb_LA$price > group_1 & airbnb_LA$price <= group_2] <- "Medium"
airbnb_LA$price_group[airbnb_LA$price > group_2] <- "High"

airbnb_LA$price_group <- factor(airbnb_LA$price_group, levels = c("Low", "Medium", "High"))

ggplot(subset(airbnb_LA, price < 1000), aes(price)) +
  geom_density(aes(fill = factor(`price_group`)), alpha = 0.5) +
  ggtitle("Listing population density by Price")

# Split the number of review into 5 parts (LR,MR,HR)
airbnb_LA <- airbnb_LA %>% 
  mutate(num.review.level = ifelse(number_of_reviews <= 10, "LR",
                                   ifelse(number_of_reviews <= 70, "MR", "HR")))
airbnb_LA$num.review.level <- factor(airbnb_LA$num.review.level, levels = c("LR", "MR", "HR"))

ggplot(subset(airbnb_LA, number_of_reviews < 500), aes(number_of_reviews)) +
  geom_density(aes(fill = factor(`num.review.level`)), alpha = 0.5) +
  ggtitle("Listing population density by Number of Reviews")

# Split the review_per_month into 4 parts (LRM, MRM, HRM)
airbnb_LA <- airbnb_LA %>% 
  mutate(review.per.month = ifelse(reviews_per_month <= 0.4, "LRM",
                                   ifelse(reviews_per_month <= 3, "MRM", "HRM")))
airbnb_LA$review.per.month <- factor(airbnb_LA$review.per.month, levels = c("LRM", "MRM", "HRM"))

ggplot(subset(airbnb_LA, reviews_per_month < 500), aes(reviews_per_month)) +
  geom_density(aes(fill = factor(`review.per.month`)), alpha = 0.5) +
  ggtitle("Listing population density by Reviews per month")

summary(airbnb_LA)


levels(airbnb_LA$price_group)
levels(airbnb_LA$review.per.month)
levels(airbnb_LA$num.review.level)
levels(airbnb_LA$room_type)
levels(airbnb_LA$neighbourhood_group)


airbnb_LA$neighbourhood_group <- factor(airbnb_LA$neighbourhood_group, levels = c("City of Los Angeles","Other Cities","Unincorporated Areas"))

# Log_Price vs Neighbourhood Group (BOX PLOT)
ggplot(airbnb_LA, aes(x = neighbourhood_group, y = log_price)) +
  geom_boxplot(aes(fill = factor(`neighbourhood_group`))) +
  th + 
  xlab("Neighbourhood Group") + 
  ylab("Log Price") +
  ggtitle("Boxplots of price by Neighbourhood Group-Airbnb_LA") +
  geom_hline(yintercept = mean(airbnb_LA$log_price), color = "purple", linetype = 2)

# BAR PLOT
ggplot(airbnb_LA, aes(neighbourhood_group)) +
  geom_bar(aes(fill = price_group)) + 
  xlab("Neighbourhood Group") + 
  ylab("Count") +
  ggtitle("Distribution of Neighbourhood")

neighbour_list <- airbnb_LA %>% 
  select(neighbourhood_group) %>% 
  dplyr::group_by(neighbourhood_group) %>% 
  dplyr::summarise(n_count = n()) %>% 
  dplyr::mutate(n_count_per = n_count * 100 / sum(n_count)) %>% 
  arrange(desc(n_count))
print(neighbour_list)

fit1 <- lm(log_price ~ neighbourhood_group, data = airbnb_LA)
summary(fit1)
hist(fit1$residuals)


# Log_Price vs Room type (BOX PLOT)
ggplot(airbnb_LA, aes(x = room_type, y = log_price)) +
  geom_boxplot(aes(fill = factor(`room_type`))) +
  th + 
  xlab("Room type") + 
  ylab("Log Price") +
  ggtitle("Boxplots of price by room type - Airbnb_LA",
          subtitle = "Entire home/apt has the highest avg price") +
  geom_hline(yintercept = mean(airbnb_LA$log_price), color = "purple", linetype = 2)

# BAR PLOT
ggplot(airbnb_LA, aes(room_type)) +
  geom_bar(aes(fill = price_group)) + 
  xlab("Room Type") + 
  ylab("Count") +
  ggtitle("Distribution of Room Type")

roomtype_list <- airbnb_LA %>% 
  select(room_type) %>% 
  dplyr::group_by(room_type) %>% 
  dplyr::summarise(n_count = n()) %>% 
  dplyr::mutate(n_count_per = n_count * 100 / sum(n_count)) %>% 
  arrange(desc(n_count))
print(roomtype_list)

fit3 <- lm(log_price ~ room_type, data = airbnb_LA)
summary(fit3)
hist(fit3$residuals)


# Price vs Number of Reviews
ggplot(airbnb_LA, aes(number_of_reviews, log_price)) +
  th + 
  theme(axis.title = element_text(), axis.title.x = element_text()) +
  geom_point(aes(color = factor(`price_group`)), alpha = 1) +
  geom_smooth(method = "lm")+
  xlab("Number of reviews") +
  ylab("Log Price") +
  ggtitle("Relationship between number of reviews vs price")

# Bar plot
review1 <- ggplot(airbnb_LA, aes(num.review.level)) +
  geom_bar(aes(fill = factor(`room_type`))) + 
  xlab("Number of Review Level") + 
  ylab("Count") +
  ggtitle("Distribution of Review")
review1

fit5 <- lm(log_price ~ num.review.level, data = airbnb_LA)
summary(fit5)
hist(fit5$residuals)



# Price VS Reviews_per_month
ggplot(airbnb_LA, aes(reviews_per_month, log_price)) +
  th +
  geom_point(alpha = 0.5, aes(color = neighbourhood_group)) +
  geom_smooth(method = "lm") +
  xlab("reviews_per_month") +
  ylab("Price") +
  ggtitle("Relationship between price and reviews_per_month") 

reviewmonth_list <- airbnb_LA %>% 
  select(reviews_per_month) %>% 
  dplyr::group_by(reviews_per_month) %>% 
  dplyr::summarise(n_count = n()) %>% 
  dplyr::mutate(n_count_per = n_count * 100 / sum(n_count)) %>% 
  arrange(desc(n_count)) 
print(reviewmonth_list)

# Bar plot
review<- ggplot(airbnb_LA, aes(review.per.month)) +
  geom_bar(aes(fill = factor(`room_type`))) + 
  xlab("Number of Reviews per month Listing was Active") + 
  ylab("Count") +
  ggtitle("Distribution of Review Per Month")

fit6 <- lm(log_price ~ review.per.month, data = airbnb_LA)
summary(fit6)
hist(fit6$residuals)

ggarrange(review1, review, nrow = 2, ncol = 1)

# Finalizing all variables
airbnb_LA <- airbnb_LA %>% 
  mutate(host.list.count = calculated_host_listings_count)
summary(airbnb_LA)

LA.Dataset <- dplyr::select(airbnb_LA, -name, -host_id, -host_name, -neighbourhood, -latitude, -longitude, -price, -minimum_nights, -number_of_reviews, -reviews_per_month, -calculated_host_listings_count)
summary(LA.Dataset)


## Part 2: Machine Learning

# Split the dataset into train/test (70:30)

LA.Dataset.1 <- dplyr::select(LA.Dataset, -price_group)
summary(LA.Dataset.1)

# Split into train/test data (70:30)
LA.Dataset.1 <- mutate(LA.Dataset.1, id = row_number())

la.train <- sample_frac(LA.Dataset.1, .7)
la.test <- anti_join(LA.Dataset.1, la.train, by = 'id')

la.train <- dplyr::select(la.train, -id)
la.test <- dplyr::select(la.test, -id)


# Forward Backward Elimination Method
## ON TRAIN DATASET
model_base.train <- lm(log_price ~ 1, data = la.train)
model_full.train <- lm(log_price ~ ., data = la.train)

# Forward selection
stepAIC(model_base.train, scope = list(upper = model_full.train, lower = model_base.train), direction = "forward", trace = FALSE)$anova

# Backward elimination
stepAIC(model_full.train, direction = "backward", trace = FALSE)$anova

# Mixed Selection starting with base model
stepAIC(model_base.train, scope = list(upper = model_full.train, lower = model_base.train), direction = "both", trace = FALSE)$anova

# Mixed Selection starting with full model
stepAIC(model_full.train, scope = list(upper = model_full.train, lower = model_base.train), direction = "both", trace = FALSE)$anova

# Final Model of Train Set from stepAIC:
#   log_price ~ neighbourhood_group + room_type + availability_365 + sqrt.min.nights + num.review.level + review.per.month

# Mutliple Linear Regression
set.seed(432)
reduced_model.train <- lm(log_price ~ neighbourhood_group + room_type + availability_365 + num.review.level + review.per.month + sqrt.min.nights + host.list.count,
                          data = la.train)
summary(reduced_model.train)

print(sprintf("Multiple R-Squared of Reduced Model of Train Set = %0.4f", (summary(reduced_model.train)$r.squared)))
print(sprintf("Adjusted R-Squared of Reduced Model of Train Set = %0.4f", (summary(reduced_model.train)$adj.r.squared)))
print(sprintf("MSE of Reduced Model of Train Set = %0.4f", anova(reduced_model.train)['Residuals', 'Mean Sq']))

# Testing accuracy
lr.pred.test <- predict(reduced_model.train, la.test)

mlr.test.mse <- mean((lr.pred.test - la.test$log_price)^2)

# "Multiple R-Squared of Reduced Model of Train Set = 0.4477"
# "Adjusted R-Squared of Reduced Model of Train Set = 0.4474"
# "MSE of Reduced Model of Train Set = 0.3330"
# Test Error Rate: 33.87%

# Decision Tree

# Variables needed for decision tree
# - neighbourhood_group
# - room_type
# - availability_365
# - price_group
# - sqrt.min.nights
# - num.review.level
# - review.per.month
# - host.list.count

LA.Dataset.2 <- dplyr::select(LA.Dataset, -log_price)

# Split into train/test data
set.seed(54321)
LA.Dataset.2 <- mutate(LA.Dataset.2, id = row_number())

la.train.DT <- sample_frac(LA.Dataset.2, .7)
la.test.DT <- anti_join(LA.Dataset.2, la.train.DT, by = 'id')

la.train.DT <- dplyr::select(la.train.DT, -id)
la.test.DT <- dplyr::select(la.test.DT, -id)

# With train data set 
fit.DT <- rpart(price_group ~ neighbourhood_group + room_type + sqrt.min.nights + num.review.level + availability_365 + review.per.month,
                data = la.train.DT)

summary(fit.DT)
rpart.plot(fit.DT, type = 4, extra = "auto", nn = TRUE)

# party package better plotting
rparty.tree <- as.party(fit.DT)
rparty.tree
plot(rparty.tree)

# Test Accuracy
DT.pred.test <- predict(fit.DT, newdata = la.test.DT, type = 'class')

# Computing Overall Error
dt.test.mse <- mean(DT.pred.test != la.test.DT$price_group)
dt.test.mse 
# TEST Error rate: 37.52 %

DT_accuracy <- confusionMatrix(DT.pred.test, la.test.DT$price_group)
DT_accuracy
# Accuracy : 0.6247

# Naive Bayes: Using caret package
set.seed(2132)
la.naive.model <- train(x = la.train.DT[-5],
                        y = la.train.DT$price_group,
                        'nb',
                        trControl = trainControl(method = 'cv', number = 10))
la.naive.model
# Prediction
la.naive.test <- predict(la.naive.model$finalModel, la.test.DT)

# Computing Overall Error
nb.test.mse <- mean(la.naive.test$class != la.test.DT$price_group)
nb.test.mse
# TEST ERROR RATE: 36.63 %

NB_accuracy <- confusionMatrix(la.naive.test$class, la.test.DT$price_group)
NB_accuracy
# Accuracy : 0.6336


# Random Forest
set.seed(543)
rf_airbnb <- randomForest(x = la.train.DT[-5],
                          y = la.train.DT$price_group,
                          ntree = 500,
                          random_state = 0)
rf_airbnb
# OOB estimate of error rate: 35.8 %

plot(rf_airbnb)
varImpPlot(rf_airbnb)
# Generate predictions
# Test Accuracy
rf.pred.test <- predict(rf_airbnb, newdata = la.test.DT)

rf.test.mse <- mean(rf.pred.test != la.test.DT$price_group)
rf.test.mse
# Test Error Rate: 35.40 %

rf_accuracy <- confusionMatrix(rf.pred.test, la.test.DT$price_group)
rf_accuracy
# Accuracy : 0.6459

# Note: Bagging is simply a special case of a random forest with m = p.

# Bagging
set.seed(193)
bag.la <- randomForest(price_group ~ neighbourhood_group + room_type + sqrt.min.nights + num.review.level + availability_365 + review.per.month + host.list.count,
                       data = la.train.DT,
                       mtry = 7,
                       importance = TRUE)
bag.la
# OOB estimate of  error rate: 40.03%

varImpPlot(bag.la)

# Prediction
yhat.bag <- predict(bag.la, newdata = la.test.DT)

bag.test.mse_6 <- mean(yhat.bag != la.test.DT$price_group)
bag.test.mse_6
# Test MSE of Bagging: 40.17113

plot(yhat.bag, la.test.DT$price_group)

bagging_accuracy_6 <- confusionMatrix(yhat.bag, la.test.DT$price_group)
bagging_accuracy_6
# Accuracy : 0.5983     

## Using 5 predictors
set.seed(5433)
bag.la.5 <- randomForest(price_group ~ neighbourhood_group + room_type + sqrt.min.nights + num.review.level + availability_365 + review.per.month + host.list.count,
                         data = la.train.DT,
                         mtry = 5,
                         importance = TRUE)
bag.la.5
#  OOB estimate of  error rate: 39.04%

varImpPlot(bag.la.5)

# Prediction
yhat.bag.5 <- predict(bag.la.5, newdata = la.test.DT)

bag.test.mse.5 <- mean(yhat.bag.5 != la.test.DT$price_group)
bag.test.mse.5
# Test MSE of Bagging: 38.73002
plot(yhat.bag.5, la.test.DT$price_group)
bagging_accurcy_5 <- confusionMatrix(yhat.bag.5, la.test.DT$price_group)
bagging_accurcy_5
# Accuracy : 0.6127

## Using sqrt(p) 
set.seed(734)
bag.la.sqrt <- randomForest(price_group ~ neighbourhood_group + room_type + sqrt.min.nights + num.review.level + availability_365 + review.per.month,
                            data = la.train.DT,
                            mtry = sqrt(7),
                            importance = TRUE)
bag.la.sqrt
# OOB estimate of  error rate: 37.42%

# Prediction
yhat.bag.sqrt <- predict(bag.la.sqrt, newdata = la.test.DT)

bag.test.mse.sq <- mean(yhat.bag.sqrt != la.test.DT$price_group)
bag.test.mse.sq
# Test MSE of Bagging: 36.88%

plot(yhat.bag.sqrt, la.test.DT$price_group)
# Test Accuracy
bagging_accurcy_sqrt <- confusionMatrix(yhat.bag.sqrt, la.test.DT$price_group)
bagging_accurcy_sqrt
# Accuracy : 0.6312

#Using the `importance()` function, we can view the importance of each variable:
importance(bag.la.sqrt)
varImpPlot(bag.la.sqrt)

# From varImpPlot() function ,we find the important variables: room_type and availability_365

# LDA 
set.seed(9545)
model_lda <- lda(price_group ~ neighbourhood_group + room_type + sqrt.min.nights + num.review.level + availability_365 + review.per.month + host.list.count,
                 data = la.train.DT)
model_lda

# Testing accuracy
lda.pred.test <- predict(model_lda, la.test.DT)

lda_accuracy <- confusionMatrix(lda.pred.test$class, la.test.DT$price_group)
lda_accuracy
# Accuracy : 0.6273

lda.test.mse <- mean(lda.pred.test$class != la.test.DT$price_group)
lda.test.mse
# 37.26638


# QDA
model_qda <- qda(price_group ~ neighbourhood_group + room_type + sqrt.min.nights + num.review.level + availability_365+ review.per.month + host.list.count,
                 data = la.train.DT)
model_qda

# Testing accuracy
qda.pred.test <- predict(model_qda, la.test.DT)

qda_accuracy <- confusionMatrix(qda.pred.test$class, la.test.DT$price_group)
qda_accuracy
# Accuracy : 0.5159

qda.test.mse <- mean(qda.pred.test$class != la.test.DT$price_group)
qda.test.mse
# 0.4841252


# kNN
LA.Dataset.3 <- dplyr::select(LA.Dataset, -log_price)
# summary(LA.Dataset.3)

# Split into train/test data (70:30)
LA.Dataset.3 <- mutate(LA.Dataset.3, id = row_number())

la.train.1 <- sample_frac(LA.Dataset.3, .7)
la.test.1 <- anti_join(LA.Dataset.3, la.train.1, by = 'id')

la.train.1 <- dplyr::select(la.train.1, -id)
la.test.1 <- dplyr::select(la.test.1, -id)
summary(la.test.1)

la.train.knn <- data.frame(as.numeric(la.train.1$neighbourhood_group),
                           as.numeric(la.train.1$room_type),
                           as.numeric(la.train.1$availability_365),
                           as.numeric(la.train.1$sqrt.min.nights),
                           as.numeric(la.train.1$num.review.level),
                           as.numeric(la.train.1$review.per.month),
                           as.numeric(la.train.1$host.list.count))

la.test.knn <- data.frame(as.numeric(la.test.1$neighbourhood_group),
                          as.numeric(la.test.1$room_type),
                          as.numeric(la.test.1$availability_365),
                          as.numeric(la.test.1$sqrt.min.nights),
                          as.numeric(la.test.1$num.review.level),
                          as.numeric(la.test.1$review.per.month),
                          as.numeric(la.test.1$host.list.count))
set.seed(1212)
# prediction k = 8
knn_predict <- knn(la.train.knn, la.test.knn, la.train.1$price_group, k = 8, prob = TRUE)
knn_Accuracy <- confusionMatrix(la.test.1$price_group, knn_predict)
knn_Accuracy
# 0.5383

knn.test.mse <- mean(knn_predict != la.test.1$price_group)
knn.test.mse
# Test error rate of KNN: 46.17%

# Accuracies Performance Comparison
DT_Accuracy <- DT_accuracy$overall[1]
NB_Accuracy <- NB_accuracy$overall[1]
RF_Accuracy <- rf_accuracy$overall[1]
Bagging_Accuracy <- bagging_accurcy_sqrt$overall[1]
LDA_Accuracy <- lda_accuracy$overall[1]
QDA_Accuracy <- qda_accuracy$overall[1]
kNN_Accuracy <- knn_Accuracy$overall[1]

Accuracy_rate <- data.frame(Model = c("Decision Tree", "Naive Bayes", "Random Forest", "Bagging", "Linear Discriminant Analysis", "Quadratic Discriminant Analysis", "k-nearest Neighbour"),
                            Accuracy = c(DT_Accuracy, NB_Accuracy, RF_Accuracy, Bagging_Accuracy, LDA_Accuracy, QDA_Accuracy, kNN_Accuracy))

Accuracy_rate$Model <- factor(Accuracy_rate$Model, 
                              levels = c("Quadratic Discriminant Analysis", "k-nearest Neighbour", "Decision Tree","Linear Discriminant Analysis", "Bagging", "Naive Bayes", "Random Forest"))

ggplot(data = Accuracy_rate, aes(x = Model, y = Accuracy, fill= Model)) + 
  geom_bar(stat = 'identity') +
  theme_bw() + 
  ggtitle('Accuracies of Models') +
  xlab("Models") +
  ylab("Accuracy") +
  coord_flip() 


# Comparing Test MSE
MLR_mse <- mlr.test.mse
DT_mse <- dt.test.mse
NB_mse <- nb.test.mse
RF_mse <- rf.test.mse
Bagging_mse <- bag.test.mse.sq
LDA_mse <- lda.test.mse
QDA_mse <- qda.test.mse
kNN_mse <- knn.test.mse

Test_MSE <- data.frame(Model = c("Decision Tree", "Naive Bayes", "Random Forest", "Bagging", "Linear Discriminant Analysis", "Quadratic Discriminant Analysis", "k-nearest Neighbour", "Multiple Linear Regression"),
                       test.mse.rate = c(DT_mse, NB_mse, RF_mse, Bagging_mse, LDA_mse, QDA_mse, kNN_mse, MLR_mse))

Test_MSE$Model <- factor(Test_MSE$Model, 
                         levels = c("Quadratic Discriminant Analysis","k-nearest Neighbour","Linear Discriminant Analysis","Decision Tree","Bagging","Naive Bayes","Random Forest","Multiple Linear Regression"))

ggplot(data = Test_MSE, aes(x = Model, y = test.mse.rate, fill= Model)) + 
  geom_bar(stat = 'identity') + 
  theme_bw() + 
  ggtitle('Test MSE of Models') +
  xlab("Models") +
  ylab("Test MSE rate") +
  coord_flip() 



# Part 3:NLP: Text Analysis

# Unigram Model
# Word Cloud - Los Angeles City
library(readr)
reviewsDec2018 <- read_csv("/System/Volumes/Data/University of Evansville/SPRING 2020/STAT 493/Airbnb_final project/RCode/Final_Coding_Khadka_files/Datasets/LA_reviewsDec18.csv")
summary(reviewsDec2018)
LA.review <- reviewsDec2018[sample(1:nrow(reviewsDec2018), 1000, replace = FALSE),]
summary(LA.review)
LA_text <- LA.review %>% 
  select(listing_id, comments)

# str(LA_text)

LA_word <- unnest_tokens(LA_text, word, comments)
LA_word <- LA_word %>% 
  anti_join(stop_words)

LA_word <- LA_word %>% 
  filter(word != "apartment", word != "location", word != "stay", word != "host", str_detect(word, "[a-z]"))

LA_word_count <- LA_word %>% 
  count(word, sort = TRUE)

LA_word_count %>% 
  with(wordcloud(word, n, max.words = 100, random.order = FALSE, colors = brewer.pal(8, "Dark2")))

# The most used words in Los Angeles are "clean", "nice", "house", "la" and "comfortable".

# word cloud - Chicago City
Dec_2018reviews <- read_csv("/System/Volumes/Data/University of Evansville/SPRING 2020/STAT 493/Airbnb_final project/RCode/Final_Coding_Khadka_files/Datasets/Chicago_reviewsDec18.csv")

Chicago.review <- Dec_2018reviews[sample(1:nrow(Dec_2018reviews), 1000, replace = FALSE),]

Chicago_text <- Chicago.review %>% 
  select(listing_id, comments)

# str(Chicago_text)

Chicago_word <- unnest_tokens(Chicago_text, word, comments)
Chicago_word <- Chicago_word %>% 
  anti_join(stop_words)

Chicago_word <- Chicago_word %>% 
  filter(word != "apartment", word != "location", word != "stay", word != "host", str_detect(word, "[a-z]"))

Chicago_word_count <- Chicago_word %>% 
  count(word, sort = TRUE)

chicago.1 <- Chicago_word_count %>% 
  with(wordcloud(word, n, max.words = 100, random.order = FALSE, colors = brewer.pal(8, "Dark2")))
# The most used words in Chicago are "clean", "chicago", "nice", "comfortable", "easy".


# word cloud - New York City
reviews <- read_csv("/System/Volumes/Data/University of Evansville/SPRING 2020/STAT 493/Airbnb_final project/RCode/Final_Coding_Khadka_files/Datasets/nyc_reviewsDec18.csv")

nyc.review <- reviews[sample(1:nrow(reviews), 1000, replace = FALSE),]

nyc_text <- nyc.review %>% 
  select(listing_id, comments)

nyc_word <- unnest_tokens(nyc_text, word, comments)
nyc_word <- nyc_word %>% 
  anti_join(stop_words)

nyc_word <- nyc_word %>% 
  filter(word != "apartment", word != "location", word != "stay", word != "host", str_detect(word, "[a-z]"))

nyc_word_count <- nyc_word %>% 
  count(word, sort = TRUE)

nyc_word_count %>% 
  with(wordcloud(word, n, max.words = 100, random.order = FALSE, colors = brewer.pal(8, "Dark2")))
# The most used words in New York City are "clean", "subway", "nice", "recommmend".

# Word Cloud - Boston
reviews <- read_csv("/System/Volumes/Data/University of Evansville/SPRING 2020/STAT 493/Airbnb_final project/RCode/Final_Coding_Khadka_files/Datasets/Boston_reviewsDec18.csv")

boston.review <- reviews[sample(1:nrow(reviews), 1000, replace = FALSE),]

boston_text <- boston.review %>% 
  select(listing_id, comments)

boston_word <- unnest_tokens(boston_text, word, comments)
boston_word <- boston_word %>% 
  anti_join(stop_words)

boston_word <- boston_word %>% 
  filter(word != "apartment", word != "location", word != "stay", word != "host", str_detect(word, "[a-z]"))

boston_word_count <- boston_word %>% 
  count(word, sort = TRUE)

boston.1 <- boston_word_count %>% 
  with(wordcloud(word, n, max.words = 100, random.order = FALSE, colors = brewer.pal(8, "Dark2")))
# The most used words in Boston City are "boston", "clean", "nice", "comfortable", "easy", and "recommend".

# Word Cloud - London
reviews_1_ <- read_csv("/System/Volumes/Data/University of Evansville/SPRING 2020/STAT 493/Airbnb_final project/RCode/Final_Coding_Khadka_files/Datasets/London_reviewsDec18.csv")

london.review <- reviews_1_[sample(1:nrow(reviews_1_), 1000, replace = FALSE),]

london_text <- london.review %>%
  select(listing_id, comments)

london_word <- unnest_tokens(london_text, word, comments)
london_word <- london_word %>% 
  anti_join(stop_words)

london_word <- london_word %>% 
  filter(word != "apartment", word != "location", word != "stay", word != "host", str_detect(word, "[a-z]"))

london_word_count <- london_word %>%
  count(word, sort = TRUE)

london_word_count %>% 
  with(wordcloud(word, n, max.words = 100, random.order = FALSE, colors = brewer.pal(8, "Dark2")))
# The most used words in London are "london", "clean", "nice", "flat", "recommmend", "lovely", "comfortable".

# word cloud - Greater Manchester
reviews <- read_csv("/System/Volumes/Data/University of Evansville/SPRING 2020/STAT 493/Airbnb_final project/RCode/Final_Coding_Khadka_files/Datasets/Manchester_reviewsDec18.csv")

manchester.review <- reviews[sample(1:nrow(reviews), 1000, replace = FALSE),]

manchester_text <- manchester.review %>% 
  select(listing_id, comments)
str(manchester_text)
manchester_word <- unnest_tokens(manchester_text, word, comments)
manchester_word <- manchester_word %>% 
  anti_join(stop_words)

manchester_word <- manchester_word %>% 
  filter(word != "apartment", word != "location", word != "stay", word != "host", str_detect(word, "[a-z]"))

manchester_word_count <- manchester_word %>% 
  count(word, sort = TRUE)

manchester_word_count %>%
  with(wordcloud(word, n, max.words = 100, random.order = FALSE, colors = brewer.pal(8, "Dark2")))
# The most used words in Greater Manchester are "clean", "manchester", recommmend", "comfortable", "lovely".

# tf-score - LA vs Chicago
LA_word_by_cty <- mutate(LA_word_count, city = "LA")
chicago_word_by_cty <- mutate(Chicago_word_count, city = "Chicago")

# Binding
chicago_LA <- bind_rows(chicago_word_by_cty, LA_word_by_cty)
word_count_by_cty <- chicago_LA  %>% 
count(city, word, sort = TRUE)

# tf score
chicago_LA_tf_idf <- chicago_LA %>% 
bind_tf_idf(word, city, n) %>% 
arrange(desc(tf_idf))

chicago_LA_tf_idf_top <- chicago_LA_tf_idf %>% 
group_by(city) %>% 
top_n(15)

ggplot(chicago_LA_tf_idf_top, aes(x = reorder(word, tf_idf), y=tf_idf, fill = city)) +
geom_bar(stat = "identity", alpha = 0.8, show.legend = FALSE) +
facet_wrap(~city, ncol = 2, scales = "free") +
scale_fill_manual(values = c("Chicago" = "blue", "LA" = "red"))+
coord_flip()


# tf-score - NYC vs Boston
nyc_word_by_cty <- mutate(nyc_word_count, city = "NYC")
boston_word_by_cty <- mutate(boston_word_count, city = "Boston")

# Binding
nyc_boston <- bind_rows(nyc_word_by_cty, boston_word_by_cty)
word_count_by_cty <- nyc_boston  %>% 
count(city, word, sort = TRUE)

# tf score
nyc_boston_tf_idf <- nyc_boston %>% 
bind_tf_idf(word, city, n) %>% 
arrange(desc(tf_idf))

nyc_boston_tf_idf_top <- nyc_boston_tf_idf %>% 
group_by(city) %>% 
top_n(15)

ggplot(nyc_boston_tf_idf_top, aes(x = reorder(word, tf_idf), y=tf_idf, fill = city)) +
geom_bar(stat = "identity", alpha = 0.8, show.legend = FALSE) +
facet_wrap(~city, ncol = 2, scales = "free") +
scale_fill_manual(values = c("NYC" = "red", "Boston" = "blue"))+
coord_flip()


# tf-score - London vs Greater Manchester
london_word_by_cty <- mutate(london_word_count, city = "London")
manchester_word_by_cty <- mutate(manchester_word_count, city = "Manchester")

# Binding
london_manchester <- bind_rows(london_word_by_cty, manchester_word_by_cty)
word_count_by_cty_2 <- london_manchester %>% 
count(city, word, sort = TRUE)

# tf score
london_manchester_tf_idf <- london_manchester %>% 
bind_tf_idf(word, city, n) %>% 
arrange(desc(tf_idf))

london_manchester_tf_idf_top <- london_manchester_tf_idf %>% 
group_by(city) %>% 
top_n(15)

ggplot(london_manchester_tf_idf_top, aes(x = reorder(word, tf_idf), y=tf_idf, fill = city)) +
geom_bar(stat = "identity", alpha = 0.8, show.legend = FALSE) +
facet_wrap(~city, ncol = 2, scales = "free") +
scale_fill_manual(values = c("London" = "green", "Manchester" = "black"))+
coord_flip()


## Bigram Model
# LA
LA_bigrams <- unnest_tokens(LA_text, bigram, comments, token = "ngrams", n = 2)
LA_bigrams %>% count(bigram, sort = TRUE)

LA_bigrams <- LA_bigrams %>%
separate(bigram, c("word1", "word2"), sep = " ") %>%
filter(!word1 %in% stop_words$word) %>% 
filter(!word2 %in% stop_words$word) %>% 
unite(bigram, word1, word2, sep = " ")

LA_bigram_count <- LA_bigrams %>% 
count(bigram, sort = TRUE) 

LA_bigram_count20 <- head(LA_bigram_count, 5) 
la.2 <- ggplot(LA_bigram_count20, aes(x = reorder(bigram, n), y = n)) + 
geom_bar(stat = "identity") + ggtitle("Bigram model of Los Angeles") + coord_flip()


# chicago
chicago_bigrams <- unnest_tokens(Chicago_text, bigram, comments, token = "ngrams", n = 2)
chicago_bigrams %>% count(bigram, sort = TRUE)

chicago_bigrams <- chicago_bigrams %>%
separate(bigram, c("word1", "word2"), sep = " ") %>%
filter(!word1 %in% stop_words$word) %>% 
filter(!word2 %in% stop_words$word) %>% 
unite(bigram, word1, word2, sep = " ")

chicago_bigram_count <- chicago_bigrams %>% count(bigram, sort = TRUE) 

chicago_bigram_count20 <- head(chicago_bigram_count, 5) 
chicago.2 <- ggplot(chicago_bigram_count20, aes(x = reorder(bigram, n), y = n)) + 
geom_bar(stat = "identity") + ggtitle("Bigram model of Chicago") + coord_flip()

ggarrange(la.2, chicago.2, nrow = 1, ncol = 2)


# New York
nyc_bigrams <- unnest_tokens(nyc_text, bigram, comments, token = "ngrams", n = 2)
nyc_bigrams %>% count(bigram, sort = TRUE)

nyc_bigrams <- nyc_bigrams %>%
separate(bigram, c("word1", "word2"), sep = " ") %>%
filter(!word1 %in% stop_words$word) %>% 
filter(!word2 %in% stop_words$word) %>% 
unite(bigram, word1, word2, sep = " ")

nyc_bigram_count <- nyc_bigrams %>% 
count(bigram, sort = TRUE) 

nyc_bigram_count20 <- head(nyc_bigram_count, 5) 

ny2<- ggplot(nyc_bigram_count20, aes(x = reorder(bigram, n), y = n)) + 
geom_bar(stat = "identity") + ggtitle("Bigram model of New York") + coord_flip()


# Boston
boston_bigrams <- unnest_tokens(boston_text, bigram, comments, token = "ngrams", n = 2)
boston_bigrams %>% count(bigram, sort = TRUE)

boston_bigrams <- boston_bigrams %>%
separate(bigram, c("word1", "word2"), sep = " ") %>%
filter(!word1 %in% stop_words$word) %>% 
filter(!word2 %in% stop_words$word) %>% 
unite(bigram, word1, word2, sep = " ")

boston_bigram_count <- boston_bigrams %>% 
count(bigram, sort = TRUE) 

boston_bigram_count20 <- head(boston_bigram_count, 5) 

bs2<- ggplot(boston_bigram_count20, aes(x = reorder(bigram, n), y = n)) + 
geom_bar(stat = "identity") + ggtitle("Bigram model of Boston") + coord_flip()

ggarrange(la.2, chicago.2, ny2, bs2, nrow = 2, ncol = 2)


# London
london_bigrams <- unnest_tokens(london_text, bigram, comments, token = "ngrams", n = 2)
london_bigrams %>% count(bigram, sort = TRUE)

london_bigrams <- london_bigrams %>%
separate(bigram, c("word1", "word2"), sep = " ") %>%
filter(!word1 %in% stop_words$word) %>% 
filter(!word2 %in% stop_words$word) %>% 
unite(bigram, word1, word2, sep = " ")

london_bigram_count <- london_bigrams %>% 
count(bigram, sort = TRUE) 

london_bigram_count20 <- head(london_bigram_count, 5) 
ln2 <- ggplot(london_bigram_count20, aes(x = reorder(bigram, n), y = n)) + 
geom_bar(stat = "identity") + ggtitle("Bigram model of London") + coord_flip()


# Manchester
manchester_bigrams <- unnest_tokens(manchester_text, bigram, comments, token = "ngrams", n = 2)
manchester_bigrams %>% count(bigram, sort = TRUE)

manchester_bigrams <- manchester_bigrams %>%
separate(bigram, c("word1", "word2"), sep = " ") %>%
filter(!word1 %in% stop_words$word) %>% 
filter(!word2 %in% stop_words$word) %>% 
unite(bigram, word1, word2, sep = " ")

manchester_bigram_count <- manchester_bigrams %>% 
count(bigram, sort = TRUE) 

manchester_bigram_count20 <- head(manchester_bigram_count, 5) 
mn2 <- ggplot(manchester_bigram_count20, aes(x = reorder(bigram, n), y = n)) + 
geom_bar(stat = "identity") + ggtitle("Bigram model of Manchester") + coord_flip()

ggarrange(la.2, chicago.2, ny2, bs2, ln2, mn2, nrow = 3, ncol = 2)


# tf-idf bigram - LA vs Chicago
LA_bigram_by_cty <- mutate(LA_bigram_count, city = "LA")
chicago_bigram_by_cty <- mutate(chicago_bigram_count, city = "Chicago")

# Binding
chicago_LA_bigram <- bind_rows(chicago_bigram_by_cty, LA_bigram_by_cty)
bigram_count_by_cty <- chicago_LA_bigram  %>%
count(city, bigram, sort = TRUE)

# tf score
chicago_LA_bigram_tf_idf <- chicago_LA_bigram %>% 
bind_tf_idf(bigram, city, n) %>% 
arrange(desc(tf_idf))

chicago_LA_bigram_tf_idf_top <- chicago_LA_bigram_tf_idf %>%
group_by(city) %>% 
top_n(5)

a1 <- ggplot(chicago_LA_bigram_tf_idf_top, aes(x = reorder(bigram, tf_idf), y=tf_idf, fill = city)) +
geom_bar(stat = "identity", alpha = 0.8, show.legend = FALSE) +
facet_wrap(~city, ncol = 2, scales = "free") +
scale_fill_manual(values = c("Chicago" = "blue", "LA" = "red"))+
coord_flip()


# tf-idf bigram - New York vs Boston
nyc_bigram_by_cty <- mutate(nyc_bigram_count, city = "NYC")
boston_bigram_by_cty <- mutate(boston_bigram_count, city = "Boston")

# Binding
nyc_boston_bigram <- bind_rows(nyc_bigram_by_cty, boston_bigram_by_cty)
bigram_count_by_cty <- nyc_boston_bigram  %>%
count(city, bigram, sort = TRUE)

# tf score
nyc_boston_bigram_tf_idf <- nyc_boston_bigram %>% 
bind_tf_idf(bigram, city, n) %>% 
arrange(desc(tf_idf))

nyc_boston_bigram_tf_idf_top <- nyc_boston_bigram_tf_idf %>%
group_by(city) %>% 
top_n(5)

a2 <- ggplot(nyc_boston_bigram_tf_idf_top, aes(x = reorder(bigram, tf_idf), y=tf_idf, fill = city)) +
geom_bar(stat = "identity", alpha = 0.8, show.legend = FALSE) +
facet_wrap(~city, ncol = 2, scales = "free") +
scale_fill_manual(values = c("Boston" = "red", "NYC" = "blue"))+
coord_flip()

ggarrange(a1, a2, nrow = 2, ncol = 1)


# tf-of bigram - London vs Manchester
london_bigram_by_cty <- mutate(london_bigram_count, city = "London")
manchester_bigram_by_cty <- mutate(manchester_bigram_count, city = "Manchester")

# Binding
london_manchester_bigram <- bind_rows(london_bigram_by_cty, manchester_bigram_by_cty)
bigram_count_by_cty <- london_manchester_bigram  %>%
count(city, bigram, sort = TRUE)

# tf score
london_manchester_bigram_tf_idf <- london_manchester_bigram %>% 
bind_tf_idf(bigram, city, n) %>% 
arrange(desc(tf_idf))

london_manchester_bigram_tf_idf_top <- london_manchester_bigram_tf_idf %>%
group_by(city) %>% 
top_n(5)

a3 <-ggplot(london_manchester_bigram_tf_idf_top, aes(x = reorder(bigram, tf_idf), y=tf_idf, fill = city)) +
geom_bar(stat = "identity", alpha = 0.8, show.legend = FALSE) +
facet_wrap(~city, ncol = 2, scales = "free") +
scale_fill_manual(values = c("London" = "green", "Manchester" = "black"))+
coord_flip()

ggarrange(a1,a2,a3, nrow = 3, ncol = 1)


## Trigram

# LA
LA_trigrams <- unnest_tokens(LA_text, trigram, comments, token = "ngrams", n = 3)
LA_trigrams %>% count(trigram, sort = TRUE)

LA_trigrams <- LA_trigrams %>%
separate(trigram, c("word1", "word2", "word3"), sep = " ") %>%
filter(!word1 %in% stop_words$word) %>%
filter(!word2 %in% stop_words$word) %>% 
filter(!word3 %in% stop_words$word) %>%
unite(trigram, word1, word2, word3, sep = " ")

LA_trigram_count <- LA_trigrams %>% 
count(trigram, sort = TRUE) 

LA_trigram_count20 <- head(LA_trigram_count, 5) 
ggplot(LA_trigram_count20, aes(x = reorder(trigram, n), y = n)) + 
geom_bar(stat = "identity") +
coord_flip()


# Chicago
chicago_trigrams <- unnest_tokens(Chicago_text, trigram, comments, token = "ngrams", n = 3)
chicago_trigrams %>% count(trigram, sort = TRUE)

chicago_trigrams <- chicago_trigrams %>%
separate(trigram, c("word1", "word2", "word3"), sep = " ") %>%
filter(!word1 %in% stop_words$word) %>%
filter(!word2 %in% stop_words$word) %>% 
filter(!word3 %in% stop_words$word) %>%
unite(trigram, word1, word2, word3, sep = " ")

chicago_trigram_count <- chicago_trigrams %>% 
count(trigram, sort = TRUE) 

chicago_trigram_count20 <- head(chicago_trigram_count, 20) 
ggplot(chicago_trigram_count20, aes(x = reorder(trigram, n), y = n)) + 
geom_bar(stat = "identity") + 
coord_flip()


# New York
nyc_trigrams <- unnest_tokens(nyc_text, trigram, comments, token = "ngrams", n = 3)
nyc_trigrams %>% count(trigram, sort = TRUE)

nyc_trigrams <- nyc_trigrams %>%
separate(trigram, c("word1", "word2", "word3"), sep = " ") %>%
filter(!word1 %in% stop_words$word) %>%
filter(!word2 %in% stop_words$word) %>% 
filter(!word3 %in% stop_words$word) %>%
unite(trigram, word1, word2, word3, sep = " ")

nyc_trigram_count <- nyc_trigrams %>% 
count(trigram, sort = TRUE) 

nyc_trigram_count20 <- head(nyc_trigram_count, 20) 
ggplot(nyc_trigram_count20, aes(x = reorder(trigram, n), y = n)) + 
geom_bar(stat = "identity") +
coord_flip()


# Boston
boston_trigrams <- unnest_tokens(boston_text, trigram, comments, token = "ngrams", n = 3)
boston_trigrams %>% count(trigram, sort = TRUE)

boston_trigrams <- boston_trigrams %>%
separate(trigram, c("word1", "word2", "word3"), sep = " ") %>%
filter(!word1 %in% stop_words$word) %>% 
filter(!word2 %in% stop_words$word) %>% 
filter(!word3 %in% stop_words$word) %>%
unite(trigram, word1, word2, word3, sep = " ")

boston_trigram_count <- boston_trigrams %>% 
count(trigram, sort = TRUE) 

boston_trigram_count20 <- head(boston_trigram_count, 20) 
ggplot(boston_trigram_count20, aes(x = reorder(trigram, n), y = n)) + 
geom_bar(stat = "identity") + 
coord_flip()


# London
london_trigrams <- unnest_tokens(london_text, trigram, comments, token = "ngrams", n = 3)
london_trigrams %>% count(trigram, sort = TRUE)

london_trigrams <- london_trigrams %>%
separate(trigram, c("word1", "word2", "word3"), sep = " ") %>%
filter(!word1 %in% stop_words$word) %>%
filter(!word2 %in% stop_words$word) %>% 
filter(!word3 %in% stop_words$word) %>%
unite(trigram, word1, word2, word3, sep = " ")

london_trigram_count <- london_trigrams %>% 
count(trigram, sort = TRUE) 

london_trigram_count20 <- head(london_trigram_count, 20) 
ggplot(london_trigram_count20, aes(x = reorder(trigram, n), y = n)) + 
geom_bar(stat = "identity") +
coord_flip()


# Manchester
manchester_trigrams <- unnest_tokens(manchester_text, trigram, comments, token = "ngrams", n = 3)
manchester_trigrams %>% count(trigram, sort = TRUE)

manchester_trigrams <- manchester_trigrams %>%
separate(trigram, c("word1", "word2", "word3"), sep = " ") %>%
filter(!word1 %in% stop_words$word) %>%
filter(!word2 %in% stop_words$word) %>% 
filter(!word3 %in% stop_words$word) %>%
unite(trigram, word1, word2, word3, sep = " ")

manchester_trigram_count <- manchester_trigrams %>% 
count(trigram, sort = TRUE) 

manchester_trigram_count20 <- head(manchester_trigram_count, 20) 
ggplot(manchester_trigram_count20, aes(x = reorder(trigram, n), y = n)) + 
geom_bar(stat = "identity") +
coord_flip()


# tf-of trigram - LA vs Chicago
LA_trigram_by_cty <- mutate(LA_trigram_count, city = "LA")
chicago_trigram_by_cty <- mutate(chicago_trigram_count, city = "Chicago")

# Binding
chicago_LA_trigram <- bind_rows(chicago_trigram_by_cty, LA_trigram_by_cty)
trigram_count_by_cty <- chicago_LA_trigram  %>%
count(city, trigram, sort = TRUE)

# tf score
chicago_LA_trigram_tf_idf <- chicago_LA_trigram %>% 
bind_tf_idf(trigram, city, n) %>% 
arrange(desc(tf_idf))

chicago_LA_trigram_tf_idf_top <- chicago_LA_trigram_tf_idf %>%
group_by(city) %>% 
top_n(10)

ggplot(chicago_LA_trigram_tf_idf_top, aes(x = reorder(trigram, tf_idf), y=tf_idf, fill = city)) +
geom_bar(stat = "identity", alpha = 0.8, show.legend = FALSE) +
facet_wrap(~city, ncol = 2, scales = "free") +
scale_fill_manual(values = c("Chicago" = "blue", "LA" = "red"))+
coord_flip()


# tf-of trigram - New York vs Boston
nyc_trigram_by_cty <- mutate(nyc_trigram_count, city = "NYC")
boston_trigram_by_cty <- mutate(boston_trigram_count, city = "Boston")

# Binding
nyc_boston_trigram <- bind_rows(nyc_trigram_by_cty, boston_trigram_by_cty)
trigram_count_by_cty <- nyc_boston_trigram  %>%
count(city, trigram, sort = TRUE)

# tf score
nyc_boston_trigram_tf_idf <- nyc_boston_trigram %>% 
bind_tf_idf(trigram, city, n) %>% 
arrange(desc(tf_idf))

nyc_boston_trigram_tf_idf_top <- nyc_boston_trigram_tf_idf %>%
group_by(city) %>% 
top_n(10)

ggplot(nyc_boston_trigram_tf_idf_top, aes(x = reorder(trigram, tf_idf), y=tf_idf, fill = city)) +
geom_bar(stat = "identity", alpha = 0.8, show.legend = FALSE) +
facet_wrap(~city, ncol = 2, scales = "free") +
scale_fill_manual(values = c("Boston" = "red", "NYC" = "blue"))+
coord_flip()
# good-value <- bon rapport qualité/prix


# tf-of trigram - London vs Manchester
london_trigram_by_cty <- mutate(london_trigram_count, city = "London")
manchester_trigram_by_cty <- mutate(manchester_trigram_count, city = "Manchester")

# Binding
london_manchester_trigram <- bind_rows(london_trigram_by_cty, manchester_trigram_by_cty)
trigram_count_by_cty <- london_manchester_trigram  %>%
count(city, trigram, sort = TRUE)

# tf score
london_manchester_trigram_tf_idf <- london_manchester_trigram %>% 
bind_tf_idf(trigram, city, n) %>% 
arrange(desc(tf_idf))

london_manchester_trigram_tf_idf_top <- london_manchester_trigram_tf_idf %>%
group_by(city) %>% 
top_n(10)

ggplot(london_manchester_trigram_tf_idf_top, aes(x = reorder(trigram, tf_idf), y=tf_idf, fill = city)) +
geom_bar(stat = "identity", alpha = 0.8, show.legend = FALSE) +
facet_wrap(~city, ncol = 2, scales = "free") +
scale_fill_manual(values = c("London" = "green", "Manchester" = "black"))+
coord_flip()


## Sentiment Analysis
#  "Sentiment analysis is one of the most obvious things we can do with unlabeled text data (with no score or no rating) to extract some insights out of it. One of the most primitive sentiment analysis is to perform a simple dictionary lookup and calculate a final composite score based on the number of occurrences of positive and negative words."

# Using 'bing' lexicon - with positive/negative annotations
# https://www.tidytextmining.com/sentiment.html

# LA
LA_sentiment <- LA_word %>% 
inner_join(get_sentiments("bing")) %>% 
count(word, sentiment, sort = TRUE) %>% 
ungroup()

LA_sentiment %>% 
group_by(sentiment) %>% 
top_n(20) %>% 
ungroup() %>% 
mutate(word = reorder(word, n)) %>% 
ggplot(aes(word, n, fill = sentiment)) +
geom_col(show.legend = FALSE) +
facet_wrap(~sentiment, scales = "free_y") + 
labs(y = "Contribution to sentiment", x = NULL) +
coord_flip()

LA_word %>% 
inner_join(get_sentiments("bing")) %>% 
count(word, sentiment, sort = TRUE) %>% 
acast(word ~ sentiment, value.var = "n", fill = 0) %>% 
comparison.cloud(colors = c("red", "blue"), max.words = 50)
# THe most positive words for LA are clean, nice, comfortable
# The most negative words for LA are die, issue, noise


# Chicago
chicago_sentiment <- Chicago_word %>% 
inner_join(get_sentiments("bing")) %>% 
count(word, sentiment, sort = TRUE) %>% 
ungroup()

chicago_sentiment %>% 
group_by(sentiment) %>% 
top_n(20) %>% 
ungroup() %>% 
mutate(word = reorder(word, n)) %>% 
ggplot(aes(word, n, fill = sentiment)) +
geom_col(show.legend = FALSE) +
facet_wrap(~sentiment, scales = "free_y") + 
labs(y = "Contribution to sentiment", x = NULL) +
coord_flip()

Chicago_word %>% 
inner_join(get_sentiments("bing")) %>% 
count(word, sentiment, sort = TRUE) %>% 
acast(word ~ sentiment, value.var = "n", fill = 0) %>% 
comparison.cloud(colors = c("red", "blue"), max.words = 50)
# THe most positive words for Chicago are clean, nice, comfortable
# The most negative words for Chicago are issues, hard, noisy


# New York
nyc_sentiment <- nyc_word %>% 
inner_join(get_sentiments("bing")) %>% 
count(word, sentiment, sort = TRUE) %>% 
ungroup()

nyc_sentiment %>% 
group_by(sentiment) %>% 
top_n(20) %>% 
ungroup() %>% 
mutate(word = reorder(word, n)) %>% 
ggplot(aes(word, n, fill = sentiment)) +
geom_col(show.legend = FALSE) +
facet_wrap(~sentiment, scales = "free_y") + 
labs(y = "Contribution to sentiment", x = NULL) +
coord_flip()

nyc_word %>% 
inner_join(get_sentiments("bing")) %>% 
count(word, sentiment, sort = TRUE) %>% 
acast(word ~ sentiment, value.var = "n", fill = 0) %>% 
comparison.cloud(colors = c("red", "blue"), max.words = 50)
# THe most positive words for New York are clean, nice, comfortable
# The most negative words for New York are die, noise, bad


# Boston
boston_sentiment <- boston_word %>% 
inner_join(get_sentiments("bing")) %>% 
count(word, sentiment, sort = TRUE) %>% 
ungroup()

boston_sentiment %>% 
group_by(sentiment) %>% 
top_n(20) %>% 
ungroup() %>% 
mutate(word = reorder(word, n)) %>% 
ggplot(aes(word, n, fill = sentiment)) +
geom_col(show.legend = FALSE) +
facet_wrap(~sentiment, scales = "free_y") + 
labs(y = "Contribution to sentiment", x = NULL) +
coord_flip()

boston_word %>% 
inner_join(get_sentiments("bing")) %>% 
count(word, sentiment, sort = TRUE) %>% 
acast(word ~ sentiment, value.var = "n", fill = 0) %>% 
comparison.cloud(colors = c("red", "blue"), max.words = 50)
# THe most positive words for Boston are clean, nice, comfortable
# The most negative words for Boston are noise, issue, bad


# London
london_sentiment <- london_word %>% 
inner_join(get_sentiments("bing")) %>% 
count(word, sentiment, sort = TRUE) %>% 
ungroup()

london_sentiment %>% 
group_by(sentiment) %>% 
top_n(20) %>% 
ungroup() %>% 
mutate(word = reorder(word, n)) %>% 
ggplot(aes(word, n, fill = sentiment)) +
geom_col(show.legend = FALSE) +
facet_wrap(~sentiment, scales = "free_y") + 
labs(y = "Contribution to sentiment", x = NULL) +
coord_flip()

london_word %>% 
inner_join(get_sentiments("bing")) %>% 
count(word, sentiment, sort = TRUE) %>% 
acast(word ~ sentiment, value.var = "n", fill = 0) %>% 
comparison.cloud(colors = c("red", "blue"), max.words = 50)
# THe most positive words for London are clean, nice, recommend
# The most negative words for London are noisy, noise, die


# Manchester
manchester_sentiment <- manchester_word %>% 
inner_join(get_sentiments("bing")) %>% 
count(word, sentiment, sort = TRUE) %>% 
ungroup()

manchester_sentiment %>% 
group_by(sentiment) %>% 
top_n(20) %>% 
ungroup() %>% 
mutate(word = reorder(word, n)) %>% 
ggplot(aes(word, n, fill = sentiment)) +
geom_col(show.legend = FALSE) +
facet_wrap(~sentiment, scales = "free_y") + 
labs(y = "Contribution to sentiment", x = NULL) +
coord_flip()

manchester_word %>% 
inner_join(get_sentiments("bing")) %>% 
count(word, sentiment, sort = TRUE) %>% 
acast(word ~ sentiment, value.var = "n", fill = 0) %>% 
comparison.cloud(colors = c("red", "blue"), max.words = 50)
# THe most positive words for Manchester are clean, nice, lovely, recommend
# The most negative words for Manchester are noise, die, issue.


## Using 'nrc' lexicon
get_sentiments("nrc")
get_sentiments("nrc") %>% count(sentiment)

# Similar positive word with 'trust' 
# The Most word associated with 'trust' seems same in all cities. Let's look at the words with a trust score from the NRC lexicon and find the most common trust words in each city dataset.
# We can see that mostly positive or good words are here about room, host and neighbourhood area.

# LA
LA_trust_nrc <- get_sentiments("nrc") %>% 
filter(sentiment == "trust")

LA_word %>% 
semi_join(LA_trust_nrc) %>% 
count(word, sort = TRUE)


# Chicago
chicago_trust_nrc <- get_sentiments("nrc") %>% 
filter(sentiment == "trust")

Chicago_word %>% 
semi_join(chicago_trust_nrc) %>% 
count(word, sort = TRUE)


# New York
nyc_trust_nrc <- get_sentiments("nrc") %>% 
filter(sentiment == "trust")

nyc_word %>% 
semi_join(nyc_trust_nrc) %>% 
count(word, sort = TRUE)


# Boston
boston_trust_nrc <- get_sentiments("nrc") %>% 
filter(sentiment == "trust")

boston_word %>% 
semi_join(boston_trust_nrc) %>% 
count(word, sort = TRUE)


# London
london_trust_nrc <- get_sentiments("nrc") %>% 
filter(sentiment == "trust")

london_word %>% 
semi_join(london_trust_nrc) %>% 
count(word, sort = TRUE)


# Manchester
manchester_trust_nrc <- get_sentiments("nrc") %>% 
filter(sentiment == "trust")

manchester_word %>% 
semi_join(manchester_trust_nrc) %>% 
count(word, sort = TRUE)


# Similar negative word with 'angry' 
# Similarly the most negative word associated with 'anger'. Let's look at the words with a anger score from the NRC lexicon and find the most common anger words in each city dataset.
# We can see that mostly negative, bad words are here about the expensive room, noisy neighbourhood area, no cooler/ AC in room. 

# LA
LA_anger_nrc <- get_sentiments("nrc") %>% 
filter(sentiment == "anger")

LA_word %>% 
semi_join(LA_anger_nrc) %>% 
count(word, sort = TRUE)


# Chicago
chicago_anger_nrc <- get_sentiments("nrc") %>% 
filter(sentiment == "anger")

Chicago_word %>% 
semi_join(chicago_anger_nrc) %>% 
count(word, sort = TRUE)


# New York
nyc_anger_nrc <- get_sentiments("nrc") %>% 
filter(sentiment == "anger")

nyc_word %>% 
semi_join(nyc_anger_nrc) %>% 
count(word, sort = TRUE)


# Boston
boston_anger_nrc <- get_sentiments("nrc") %>% 
filter(sentiment == "anger")

boston_word %>% 
semi_join(boston_anger_nrc) %>% 
count(word, sort = TRUE)


# London
london_anger_nrc <- get_sentiments("nrc") %>% 
filter(sentiment == "anger")

london_word %>% 
semi_join(london_anger_nrc) %>% 
count(word, sort = TRUE)


# Manchester
manchester_anger_nrc <- get_sentiments("nrc") %>% 
filter(sentiment == "anger")

manchester_word %>% 
semi_join(manchester_anger_nrc) %>% 
count(word, sort = TRUE)