library(tibble)
library(readr)
data_dir <-"~/Downloads/231-final/231-RNN/data"
fname <- file.path(data_dir,"data_clean.RData")
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
#plot(data[500:1000, 2], type = 'l')
##################
### calling generator function
source('~/Downloads/231-final/231-RNN/scripts/generator.R')
lookback <- 5
step <- 1
delay <- 0
batch_size <- 128
#set seed
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
# Adjusting the generator function to include only the past 5 days' stock prices
# generator_5days <- function(data,
#                             lookback,
#                             delay,
#                             min_index,
#                             max_index,
#                             shuffle = FALSE,
#                             batch_size = 128,
#                             step = 1) {
#   if (is.null(max_index))
#     max_index <- nrow(data) - delay - 1
#   i <- min_index + lookback
#   function() {
#     if (shuffle) {
#       rows <- sample(c((min_index + lookback):max_index), size = batch_size)
#     } else {
#       if (i + batch_size >= max_index)
#         i <<- min_index + lookback
#       rows <- c(i:min(i + batch_size - 1, max_index))
#       i <<- i + length(rows)
#     }
#     samples <- array(0, dim = c(length(rows),
#                                 lookback / step * ncol(data),
#                                 1))  # Adjusting to include only one feature (past stock prices)
#     targets <- array(0, dim = c(length(rows)))
#     
#     for (j in 1:length(rows)) {
#       indices <- seq(rows[[j]] - lookback, rows[[j]] - 1,
#                      length.out = dim(samples)[[2]])
#       samples[j, ,1] <- data[indices, 5]  # Assuming the 5th column represents the stock prices
#       targets[[j]] <- data[rows[[j]] + delay, 5]
#     }
#     list(samples, targets)
#   }
# }

# Creating a new generator using only past 5 days' stock prices
source('~/Downloads/231-final/231-RNN/scripts/generator.R')
lookback <- 5
step <- 1
delay <- 0
batch_size <- 128
set.seed(123)
train_gen_5days <- generator_5days(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 500,
  max_index = 1000,
  shuffle = FALSE,
  step = step,
  batch_size = batch_size
)

train_gen_data_5days <- train_gen_5days() 


model <- keras_model_sequential() %>%
  layer_flatten(input_shape = c(lookback / step , 6)) %>%  # Flatten layer to handle the input shape
  layer_dense(units = 32, activation = 'relu') %>%  # Dense layer with ReLU activation
  layer_dense(units = 1)  # Output layer with 1 unit

model %>% compile(optimizer = 'adam', loss = 'mae')


history <- model %>% fit(
  x = samples,
  y = targets,
  epochs = 30,
  batch_size = 32,  # Adjust batch size based on your resources
  validation_split = 0.2
)

plot(history)

