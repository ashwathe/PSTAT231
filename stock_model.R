library(tidyverse)
library(tidymodels)
library(tidytext)
library(keras)
library(tensorflow)
library(yardstick)
library(ggplot2)
library(dplyr)
setwd("~/231-RNN/data/")
load("data_clean.RData")
getwd()
# Looking at the data
ggplot(data_clean, aes(x = 1:nrow(data_clean), y = Close)) + geom_line()

# Looking at the portion we're potentially interested in for training
ggplot(data_clean[(500:1000),], aes(x = 500:1000, y = Close)) + geom_line()

### Standardize and Normalize data
data <- data.matrix(data_clean[,-1])

# Standardize data --> center around mean for each column
#train_data <- data[(500:1000),]
mean <- apply(data, 2, mean)
std <- apply(data, 2, sd)
data <- scale(data, center = mean, scale = std)

# Normalize, create func. --> make between 0 and 1 for activation function 
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

max <- apply(data, 2, max)
min <- apply(data, 2, min)

# Normalize data & get rid of adjusted close 
data <- apply(data, 2, normalize)

# Shape of standardized, normalized data is the same as before
plot(data[500:1000, 2], type = 'l')

### calling generator function
source('~/Downloads/231-final/231-RNN/scripts/generator.R')
lookback <- 5
step <- 1
delay <- 0
batch_size <- 128

set.seed(123)
train_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 500,
  max_index = 1000,
  shuffle = FALSE,
  step = step,
  batch_size = batch_size)

train_gen_data <- train_gen() 

### Setting up model
model <- keras_model_sequential() %>%
  layer_lstm(units = 64, input_shape = c(lookback, dim(data)[-1]))  %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1)


model %>% compile(loss = 'mean_squared_error', optimizer = 'adam',metrics='mse')

history <-
  model %>% fit (
    train_gen_data[[1]],train_gen_data[[2]],
    batch_size = 128,
    epochs = 50,
    validation_split = 0.1,
    use_multiprocessing = T
  )

plot(history)

# Plotting actual test data vs. predicted 
batch_size_plot <- 128
lookback_plot <- 5
step_plot <- 1

set.seed(123)
pred_gen <- generator(
  data,
  lookback = lookback_plot,
  delay = 0,
  min_index = 1000,
  max_index = 1260,
  shuffle = FALSE,
  step = step_plot,
  batch_size = batch_size_plot
)

pred_gen_data <- pred_gen()

V1 = seq(1, length(pred_gen_data[[2]]))

# binds V1 as time step (actual) to actual sequence 
plot_data <- as.data.frame(cbind(V1, pred_gen_data[[2]]))

inputdata <- pred_gen_data[[1]][,,]
dim(inputdata) <- c(batch_size_plot,lookback_plot, 6)

pred_out <- model %>% predict(inputdata) 

plot_data <- cbind(plot_data, pred_out[])

p <- ggplot(plot_data, aes(x = V1, y = V2)) + geom_line( colour = "blue", size = 1, alpha=0.4)
p <- p + geom_line(aes(x = V1, y = pred_out), colour = "red", size = 1 , alpha=0.4)

p

### Model that uses only past 5 days stock prices
source('~/Downloads/231-final/231-RNN/scripts/generator.R')
lookback <- 5
step <- 1
delay <- 0
batch_size <- 128

train_gen <- generator_1v(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 500,
  max_index = 1000,
  shuffle = FALSE,
  step = step,
  batch_size = batch_size)

train_gen_data <- train_gen() 

### Setting up model//
model <- keras_model_sequential() %>%
  layer_lstm(units = 64, input_shape = c(lookback, dim(data)[-1]))  %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1)


model %>% compile(loss = 'mean_squared_error', optimizer = 'adam',metrics='mse')

history <-
  model %>% fit (
    train_gen_data[[1]],train_gen_data[[2]],
    batch_size = 128,
    epochs = 50,
    validation_split = 0.1,
    use_multiprocessing = T
  )

plot(history)



