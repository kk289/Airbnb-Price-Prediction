# Senior Seminar Project
# Author: Kevil Khadka

# Title:  Inside Airbnb Using Machine Learning and R concept

# loading libraries
library(tidyverse)
library(dplyr)
library(caret)
library(MASS)
library(mice)
library(GGally)
library(leaps)
library(caTools)
library(ggthemes)
library(ggExtra)
library(glmnet)
library(corrplot)
library(leaflet)
library(plotly)
library(rpart)
library(rpart.plot)
setwd("/System/Volumes/Data/University of Evansville/SPRING 2020/STAT 493/Airbnb_final project/los angeles/05 Dec 2019")

## Importing dataset
set.seed(123)
airbnb_LA <- read_csv("listings.csv") %>% 
  filter(airbnb_LA$price>0)
summary(airbnb_LA)

## Structure of dataset
str(airbnb_LA)

airbnb_LA <- read_csv("listings.csv", 
                      col_types = cols(id = col_double(),
                                       name = col_character(),
                                       host_id = col_double(),
                                       host_name = col_factor(),
                                       neighbourhood_group = col_factor(),
                                       neighbourhood = col_factor(),
                                       latitude = col_double(),
                                       longitude = col_double(),
                                       room_type = col_factor(),
                                       price = col_double(),
                                       minimum_nights = col_double(),
                                       number_of_reviews = col_double(),
                                       last_review = col_date(format = ""),
                                       reviews_per_month = col_double(),
                                       calculated_host_listings_count = col_double(),
                                       availability_365 = col_double()
                      ))

glimpse(airbnb_LA)
head(airbnb_LA)

## Data Exploration
dim(airbnb_LA)
head(airbnb_LA, 6)
tail(airbnb_LA, 6)

# Missing Data
contents(airbnb_LA)
is.na(airbnb_LA)

missing_airbnb <- airbnb_LA %>% summarise_all(~(sum(is.na(.))/n()))
missing_airbnb <- gather(missing_airbnb, key = "variables", value = "percent_missing")
missing_airbnb <- missing_airbnb[missing_airbnb$percent_missing > 0.0, ]

theme_ms <- theme_fivethirtyeight() + theme(axis.title = element_text(), axis.title.x = element_text())

ggplot(missing_airbnb, aes(x = reorder(variables, percent_missing), y = percent_missing)) + 
  geom_bar(stat = "identity", fill = "red", aes(color = I('white')), size = 0.3) +
  xlab('variables') +
  coord_flip() +
  theme_ms +
  ggtitle("Missing Data") +
  xlab("Column name") +
  ylab("Percentage missing") +
  annotate("text", x = 1.5, y = 0.1, label = "name has less than 0.001\n percentage missing", color = "slateblue", size = 5)

# The MICE package also has a table for us to focus on the `NA` values.
md.pattern(airbnb_LA)


# data cleaning
# changing NA value to 0
airbnb_LA$reviews_per_month[is.na(airbnb_LA$reviews_per_month)] <- 0
airbnb_LA$calculated_host_listings_count[is.na(airbnb_LA$calculated_host_listings_count)] <- 0


# Leaflet map
pal <- colorFactor(palette = c("red", "green", "blue", "purple", "yellow"), domain = airbnb_LA$neighbourhood_group)

leaflet(data = airbnb_LA) %>% 
  addProviderTiles(providers$CartoDB.DarkMatterNoLabels) %>% 
  addCircleMarkers(~longitude, ~latitude, color = ~pal(neighbourhood_group),
                   weight = 1, radius=1, fillOpacity = 0.1, opacity = 0.1,
                   label = paste("Name:", airbnb_LA$name)) %>% 
  addLegend("bottomright", pal = pal, values = ~neighbourhood_group,
            title = "Neighbourhood groups", opacity = 1)


## Machine Learning

# split 'price' into 3 levels

data_temp <- sort.int(airbnb_LA$price, decreasing = FALSE)

level_1 <- data_temp[round(length(data_temp)/3, digits = 0)]
level_2 <- data_temp[2*round(length(data_temp)/3, digits = 0)]

airbnb_LA$price_level[airbnb_LA$price <= level_1] <- "Low"
airbnb_LA$price_level[airbnb_LA$price > level_1 & airbnb_LA$price <= level_2] <- "Medium"
airbnb_LA$price_level[airbnb_LA$price > level_2] <- "High"

airbnb_LA$price_level <- as.factor(airbnb_LA$price_level)

## Splitting dataset into train/test data for decision tree

airbnb_LA_new <- airbnb_LA %>% 
  select(c(price_level, neighbourhood, room_type, minimum_nights)) %>% 
  na.omit()

create_train_test <- function(data, size = 0.75, train = TRUE){
  n_row = nrow(data)
  total_row = size * n_row
  train_sample <- (1:total_row)
  if (train == TRUE){
    return (data[train_sample, ])
  } else {
    return (data[-train_sample, ])
  }
}

# testing function
train_data <- create_train_test(airbnb_LA_new, 0.75, train = TRUE)
test_data <- create_train_test(airbnb_LA_new, 0.75, train = FALSE)

#checking
nrow(train_data) + nrow(test_data) == nrow(airbnb_LA_new)

dim(train_data)
dim(test_data)

## To verify if the randomization process is correct.
# Use the function prop.table() combined with table() to verify if the randomization process is correct.

prop.table(table(train_data$price_level))
prop.table(table(test_data$price_level))

# In both dataset, the amount of low price_level is the same, about 33 percent,

## Build the model: Decision Tree

fit <- rpart(price_level ~ ., data = train_data, method = 'class', control = rpart.control(cp = 0.05))
rpart.plot(fit, type = 4, extra = "auto", nn = TRUE)

# predict the price using test data
pred <- predict(fit, newdata = test_data, type = 'response')
str(pred)

# confusion matrix
confusionMatrix(pred$class, test_data$price_level, positive = "low")

